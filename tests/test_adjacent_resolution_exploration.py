"""Small local mathematical checks; no 64²/128² or GPU benchmark is implied."""
from types import SimpleNamespace
import json

import pytest
import torch

from tdn.analysis.adjacent import resolution_exploration as e
from tdn.analysis.roadmap.numerics import quadratic_df_defect
from tdn.numerics import Equation,Geometry


def _field():
    x=torch.arange(8,dtype=torch.float32)/8
    xx,yy=torch.meshgrid(x,x,indexing="ij")
    return (.43+.05*torch.cos(2*torch.pi*(xx+yy))+.03*torch.sin(4*torch.pi*xx))[None,None]


def test_interaction_selector_does_not_evaluate_skipped_cubic():
    u=_field();eq=Equation(.004,3.);geo=Geometry((8,8),(1.,1.))
    def forbidden(*args,**kwargs):raise AssertionError("Skipped cubic was computed")
    result,features=e.selected_step(u,.04,eq,geo,"discrete",target=1.,cubic_function=forbidden)
    assert not features["cubic_computed"]
    torch.testing.assert_close(result,e.direct_step(u,.04,eq,geo,"discrete"))
    _,features=e.selected_step(u,.04,eq,geo,"discrete",target=1e-20)
    assert features["cubic_computed"]


@pytest.mark.parametrize("track",["discrete","continuum"])
def test_device_encoding_null_precision_and_off_node_parity(track):
    u=_field();eq=Equation(.004,3.);geo=Geometry((8,8),(1.,1.))
    encoded=e.DeviceCorrectionEncoding(u,eq,geo,track,.04,degree=6)
    assert encoded.coefficients.dtype==torch.float32
    assert encoded.preparation_precision=="float64"
    torch.testing.assert_close(encoded.query(0.),u,atol=0,rtol=0)
    for h in (.005,.017,.04):
        expected=quadratic_df_defect(u.double(),h,eq,geo,target=track,nodes=4).float()
        torch.testing.assert_close(encoded.correction(h),expected,atol=5e-11,rtol=5e-4)
    with pytest.raises(ValueError,match="interval"):encoded.query(.041)
    with pytest.raises(ValueError,match="refresh"):
        encoded.validate(u+.001,eq,geo,track,.04)
    with pytest.raises(ValueError,match="refresh"):
        encoded.validate(u,Equation(.006,3.),geo,track,.04)


def test_accuracy_gate_requires_accepted_reference_and_both_norms():
    reference=torch.zeros(1,1,4,4,dtype=torch.float64)
    prediction=reference.float();prediction[0,0,0,0]=1e-4
    error=e._errors(prediction,reference,dict(accepted=True,uncertainty_rms=1e-12,uncertainty_max_bound=1e-12))
    assert error["error_rms"]<3e-5 and not e._qualified(error,3e-5)
    error=e._errors(reference,reference,dict(accepted=False,uncertainty_rms=1e-12,uncertainty_max_bound=1e-12))
    assert not e._qualified(error,3e-5)


@pytest.mark.parametrize("track",["discrete","continuum"])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_cached_analytic_quadrature_is_same_formula_and_reuses_only_operator(track,dtype):
    from tdn.analysis.frontier.numerics import CoefficientCache
    u=_field().to(dtype);eq=Equation(.004,3.);geo=Geometry((8,8),(1.,1.));cache=CoefficientCache()
    for state in (u,u+.02):
        actual=e.cached_quadratic(state,.04,eq,geo,track,cache)
        expected=quadratic_df_defect(state,.04,eq,geo,nodes=4,target=track)
        torch.testing.assert_close(actual,expected,rtol=2e-4 if dtype==torch.float32 else 1e-8,
                                   atol=3e-10 if dtype==torch.float32 else 1e-17)
    assert cache.hits>0


@pytest.mark.parametrize("speedup",[.5,2.])
def test_missing_interior_accuracy_is_na_whether_query_timing_is_fast_or_slow(speedup):
    measured=[dict(speedup_over_best_endpoint_qualified=speedup,finite=True,
                   reference_accepted=True,endpoint_qualified=True)]
    result=e.temporal_group_decision(8,measured,1,1.1)
    assert result["outcome"]=="NA"
    assert result["descriptive_cost_margin_failed"] is (speedup<1.1)
    assert result["endpoint_only_outcome"]==("BAD" if speedup<1.1 else "GOOD")
    endpoint=e.temporal_group_decision(1,measured,1,1.1)
    assert endpoint["outcome"]==("BAD" if speedup<1.1 else "GOOD")


def test_known_endpoint_failure_stays_visible_without_claiming_interior_validation():
    measured=[dict(speedup_over_best_endpoint_qualified=2.,finite=True,
                   reference_accepted=True,endpoint_qualified=False)]
    result=e.temporal_group_decision(8,measured,1,1.1)
    assert result["outcome"]=="NA" and result["endpoint_only_outcome"]=="BAD"
    assert result["endpoint_accuracy_failure"]


class _Budget:
    def check(self):pass


def test_resolution_run_consumes_bank_physics_and_keeps_missing_interior_truth_na(tmp_path,monkeypatch):
    from tdn.analysis.adjacent import resolution_diagnostics as bank
    # A small synthetic bank fixture isolates dispatch/cost semantics. It is
    # deliberately not advertised as a measured native high-resolution bank.
    u=torch.full((1,1,4,4),.43,dtype=torch.float64);h=.04;rate=2.
    target=u*torch.exp(torch.tensor(rate*h))/(1+u*torch.expm1(torch.tensor(rate*h)))
    metadata=dict(parent_id="fixture-parent",parent=dict(kappa=.009,reaction_rate=rate,domain=[2.,1.]),
        accepted=True,reference_accepted=True,uncertainty_rms=1e-8,uncertainty_max_bound=1e-8)
    monkeypatch.setattr(bank,"iterate_bank_entries",lambda path:[metadata])
    monkeypatch.setattr(bank,"load_bank_entry",lambda *args:(u,target,metadata))
    seen=[]
    protocol=dict(resolution=dict(tolerances=[2e-5],training_horizons=[h],seed=91,
        exploration=dict(parent_limit=1,query_counts=[1,4],repeats=1,temporal_degree=3)))
    ctx=SimpleNamespace(protocol=protocol,unit=dict(grid=4,track="discrete",seconds=30,bank_units=["bank"]),
        path=tmp_path,prerequisites={"bank":tmp_path/"fixture-bank"},device="cpu",budget=_Budget(),
        record=lambda *args,**kwargs:seen.append((args,kwargs)))
    result=e.run(ctx)
    assert result["evidence_status"]=="COMPLETE_BOUNDED_DEVELOPMENT"
    assert result["temporal"]["advance"] is False
    rows=json.loads((tmp_path/"resolution_exploration_rows.json").read_text())
    assert {r["mechanism_id"] for r in rows}=={"HR-E01","HR-E02"}
    assert all(r["config"]["kappa"]==.009 and r["config"]["reaction_rate"]==rate for r in rows)
    assert all(r["metrics"]["all_query_accuracy"] is None for r in rows if r["metrics"].get("query_count")==4)
    costs=json.loads((tmp_path/"resolution_exploration_costs.json").read_text())
    refresh=[r for r in costs if r["variant"]=="autonomous_refresh"]
    assert refresh and all(r["timing"]["raw"] for r in refresh)
    scope=json.loads((tmp_path/"resolution_exploration_scope.json").read_text())
    assert scope["references_regenerated"] is False
    assert result["parent_denominator"]==1 and seen
