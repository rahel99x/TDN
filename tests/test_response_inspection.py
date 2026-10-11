"""Read-only source-factor inspection with honest zero-rank and missing cases."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tdn.analysis.response import inspection
from tdn.analysis.response.models import create_model
from tdn.numerics.types import Geometry
from tdn.research.experiment import Budget


def sample():
    x = torch.arange(8)/8
    u = (.3+.05*torch.cos(2*torch.pi*x[:, None])+.02*torch.sin(4*torch.pi*x[None, :]))[None, None]
    return dict(u0=u, record=dict(parent_id="declared-first", query_id="declared-first/h", target_kind="fd_same_grid",
        grid=[8,8], lengths=[1.,1.], horizon=.06, kappa=.003, reaction_rate=3., input_sha256="input"))


def model():
    return create_model(dict(family="coupled_source", width=3, depth=1, modes=1, rank=2,
        normalization=dict(mean_scale=.001, centered_scale=.001, full_scale=.1, full_mean=.4, floor=1e-7)))


@pytest.fixture(autouse=True)
def one_thread():
    count = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


def test_zero_readout_has_nonzero_basis_but_zero_active_source():
    network = model(); network.train()
    before = deepcopy(network.state_dict())
    row, arrays = inspection.inspect_model(network, sample(), fractions=(0., .5, 1.))
    assert row["zero_active_source"]
    assert row["source_field_rank"]["numerical_rank"] > 0
    assert row["active_final_rank"]["numerical_rank"] == 0
    assert row["observed_source_time_rank"]["effective_rank_entropy"] == 0
    assert np.count_nonzero(arrays["combined_source"]) == 0
    assert network.training
    assert all(torch.equal(value, network.state_dict()[key]) for key,value in before.items())
    assert not row["teacher_access"]


def test_nonzero_source_reports_actual_coefficients_response_and_safe_arrays(tmp_path):
    network = model()
    with torch.no_grad():
        network.source_head.weight.fill_(.15)
        network.source_head.bias.copy_(torch.tensor([.2,-.1]))
        network.mean_head.bias.fill_(.05)
    method = lambda sample: None
    method.model = network
    method.metadata = dict(family="coupled_source", method_id="coupled_source::primary::n32", seed=1,
                           variant_id="primary", checkpoint_path="frozen/model", selection_sha256="selection")
    ctx = SimpleNamespace(path=tmp_path, protocol={}, budget=Budget(30))
    result = inspection.run(ctx, [sample()], {"model/seed-1":method})
    row = result["rows"][0]
    assert row["status"] == "COMPLETED"
    assert not row["zero_active_source"]
    assert row["active_final_rank"]["numerical_rank"] > 0
    assert row["times"][0]["source_rms"] == 0
    assert row["times"][-1]["source_rms"] > 0
    assert row["times"][-1]["full_correction_rms"] > 0
    assert row["maximum_source_reconstruction_difference"] < 1e-8
    assert "ambiguity" in row["interpretation"]
    with np.load(tmp_path/row["arrays"], allow_pickle=False) as arrays:
        assert arrays["source_fields"].shape == (1,2,8,8)
        assert arrays["combined_source"].shape == (5,1,1,8,8)
        assert all(arrays[name].dtype != object for name in arrays.files)
    assert json.loads((tmp_path/"source_diagnostics.json").read_text())["rows"][0]["arrays_sha256"] == row["arrays_sha256"]


def test_missing_prototype_is_explicit_and_budget_censoring_retains_denominator(tmp_path):
    ctx = SimpleNamespace(path=tmp_path, protocol={}, budget=Budget(30))
    result = inspection.run(ctx, [sample()], {})
    assert result["rows"][0]["status"] == "NOT_MEASURED"
    assert result["expected_models"] == []
    method = lambda _:None; method.model=model(); method.metadata={}
    ctx.protocol={"response":{"inspection":{"seconds":0}}}
    result=inspection.run(ctx,[sample()],{"declared":method})
    assert result["expected_models"] == ["declared"]
    assert result["rows"][0]["status"] == "CENSORED"


def test_physical_rank_handles_nyquist_corner_and_zero_energy():
    alternating=(-1.)**torch.arange(8, dtype=torch.float64)
    mode=(alternating[:,None]*alternating[None,:])[None,None]
    fields=torch.stack([mode, 2*mode])
    rank=inspection.field_rank(fields,Geometry((8,8),(1.,1.)),"galerkin_same_grid")
    assert rank["numerical_rank"] == 1
    assert rank["total_physical_energy"] == pytest.approx(1.25)
    assert rank["stable_rank"] == pytest.approx(1.)
    assert inspection.field_rank(fields*0,Geometry((8,8),(1.,1.)),"galerkin_same_grid")["numerical_rank"] == 0


@pytest.mark.parametrize("representation,learned",[("learned_basis",True),("fixed_physical",False),("direct_interpolation",False)])
def test_all_source_representations_use_the_actual_shared_components(representation, learned):
    network = create_model(dict(family="coupled_source", width=3, depth=1, modes=1, rank=2,
        source_representation=representation, learned_time_rates=learned))
    with torch.no_grad():
        if representation == "direct_interpolation":
            network.interpolation_head.weight.fill_(.05)
        else:
            network.source_head.bias.fill_(.2)
    row, arrays = inspection.inspect_model(network,sample(),fractions=(0.,.4,1.))
    assert row["source_representation"] == representation
    assert row["maximum_source_reconstruction_difference"] == 0
    assert row["times"][-1]["source_rms"] > 0
    assert sum(sum(band["channel_energy"]) for band in row["source_field_spectral_bands"]) == pytest.approx(
        row["source_field_rank"]["total_physical_energy"], rel=1e-10)


def clock_model():
    network = create_model(dict(family="coupled_source", width=3, depth=1, modes=1, rank=2,
                                mean_parameterization="clock"))
    with torch.no_grad():
        network.mean_head.bias.copy_(torch.tensor([.4,-.3,.2,.1]))
        network.source_head.bias.fill_(.3)
    return network


def test_clock_implied_variance_is_measured_not_assumed_equal_to_physical_variance():
    item = sample(); network = clock_model()
    row, _ = inspection.inspect_model(network,item,fractions=(0., .5, 1.))
    zero, middle, final = row["times"]
    assert zero["clock_variance_status"] == "MEASURED"
    assert zero["reconstructed_mean_derivative_method"] == "second-order finite difference with step halving"
    assert final["clock_implied_variance"] == pytest.approx(
        final["learned_mean"]*(1-final["learned_mean"])-final["mean_path_derivative"]/3.)
    assert final["clock_implied_minus_physical_variance"] == pytest.approx(
        final["clock_implied_variance"]-final["reconstructed_physical_variance"])
    assert abs(final["clock_implied_minus_physical_variance"]) > 1e-6
    assert final["mean_path_ode_residual"] == pytest.approx(
        -3*final["clock_implied_minus_physical_variance"], abs=1e-12)
    assert final["mean_ode_residual"] == pytest.approx(final["reconstructed_mean_derivative"]-3*(
        final["prediction_mean"]*(1-final["prediction_mean"])-final["reconstructed_physical_variance"]))
    assert "penalty, not an identity" in final["mean_ode_residual_scope"]
    assert all(parameter.grad is None for parameter in network.parameters())


def test_clock_variance_na_for_zero_reaction_and_invalid_reconstructed_bounds():
    item = sample(); item["record"]["reaction_rate"] = 0.
    row,_ = inspection.inspect_model(clock_model(),item,fractions=(1.,))
    endpoint=row["times"][0]
    assert endpoint["clock_variance_status"] == "NA"
    assert endpoint["clock_implied_variance"] is None
    assert "positive reaction" in endpoint["clock_variance_reason"]
    network=clock_model(); item=sample()
    eq,geom=inspection.equation_geometry(item["record"])
    with torch.no_grad():
        encoded=network.encode(item["u0"],eq,geom,"fd_same_grid")
        invalid=network.decode(encoded,.06)+2.
        endpoint=inspection.mean_balance(network,encoded,.06,invalid)
    assert endpoint["clock_variance_status"] == "NA"
    assert endpoint["clock_implied_variance"] is None
    assert "bounded" in endpoint["clock_variance_reason"]


def test_clock_derivative_fallback_reports_halving_sensitivity(monkeypatch):
    network=clock_model(); item=sample()
    eq,geom=inspection.equation_geometry(item["record"])
    with torch.no_grad():
        encoded=network.encode(item["u0"],eq,geom,"fd_same_grid")
        prediction=network.decode(encoded,.06)
        original=inspection.mean_balance(network,encoded,.06,prediction)
    def unsupported(*args,**kwargs):
        raise RuntimeError("Test unavailable autodifferentiation")
    monkeypatch.setattr(torch.autograd,"grad",unsupported)
    with torch.no_grad():
        fallback=inspection.mean_balance(network,encoded,.06,prediction)
    assert fallback["mean_derivative_method"] == "second-order finite difference with step halving"
    assert fallback["mean_derivative_halving_difference"] >= 0
    assert fallback["mean_path_derivative"] == pytest.approx(original["mean_path_derivative"],abs=4e-4)
