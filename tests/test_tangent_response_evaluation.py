import json
import math

import pytest
import torch

from tdn.analysis.tangent_response.contracts import Geometry, Physics
from tdn.analysis.tangent_response.evaluation import (
    assert_parent_disjoint, evaluate_endpoints, evaluate_rollout,
    nonlinear_endpoint_composition, physical_metrics, run,
    same_path_tangent_composition, sensitivity_panel, finite_product_constant,
    conditional_response_bound, signed_error_decomposition,
)


def sample(parent="p",split="diagnostic",grid=5,target="fd_same_grid",accepted=True,value=.2):
    u=torch.full((1,1,grid,grid),value,dtype=torch.float32)
    return dict(u0=u,teacher64=u.double(),accepted=accepted,
                reference=dict(accepted=accepted,uncertainty_rms=1e-12,uncertainty_maximum=1e-12),
                record=dict(parent_id=parent,query_id=parent+"-q",split=split,target_kind=target,
                            grid=[grid,grid],lengths=[1.,1.],horizon=.12,kappa=.01,reaction_rate=.2,
                            shape_sha256=parent+"-shape",regime="mid_mean"))


@pytest.mark.parametrize("grid",[5,6])
@pytest.mark.parametrize("target",["fd_same_grid","galerkin_same_grid"])
def test_full_calls_physical_metrics_and_json_ready(grid,target):
    calls=[]
    def model(u,h,eq,geom,kind):
        calls.append((u,h,eq,geom,kind))
        return u+.01
    s=sample(grid=grid,target=target)
    result=run([s],{"control":model},{})
    row=result["endpoints"]["control"]["rows"][0]
    assert row["status"]=="MEASURED"
    assert row["rms"]==pytest.approx(.01,abs=1e-8)
    assert row["mean_mse"]==pytest.approx(row["mse"])
    assert row["centered_mse"]<1e-20
    assert len(calls)==1 and len(calls[0])==5
    json.dumps(result,allow_nan=False)


def test_even_nyquist_uses_physical_mass_and_sampled_maximum():
    grid=6
    x=torch.arange(grid,dtype=torch.float64)
    error=torch.cos(math.pi*x).reshape(1,1,grid,1).expand(1,1,grid,grid)
    geom=Geometry((grid,grid),(1.,1.))
    fd=physical_metrics(error,geom,"fd_same_grid")
    gal=physical_metrics(error,geom,"galerkin_same_grid")
    assert fd["mse"]==pytest.approx(1.)
    assert gal["mse"]==pytest.approx(.5)
    assert gal["maximum_grid"]==[12,12]
    assert "not a continuous supremum" in gal["maximum_scope"]


def test_failures_unresolved_and_unsupported_stay_in_denominator():
    s=[sample("accepted"),sample("unresolved",accepted=False),sample("fine",target="galerkin_projected_fine")]
    result=evaluate_endpoints(lambda u,*args:torch.full_like(u,float("nan")),s)
    assert [r["status"] for r in result["rows"]]==["FAILED_NONFINITE","UNRESOLVED_REFERENCE","UNSUPPORTED_TARGET"]
    summary=result["summaries"]["diagnostic/fd_same_grid"]
    assert summary["requested_queries"]==2 and summary["measured_queries"]==0


def test_shape_and_parent_split_leakage_rejected():
    a,b=sample("a",split="train"),sample("b",split="validation")
    b["record"]["shape_sha256"]=a["record"]["shape_sha256"]
    with pytest.raises(ValueError,match="leakage"):
        assert_parent_disjoint([a,b])


def test_rollout_reencodes_actual_state_and_teachers_original_input():
    s=sample(value=.99)
    calls,origins=[],[]
    def model(u,h,*args):
        calls.append(u.clone())
        return u+h
    def provider(original,at):
        origins.append(original["u0"].clone())
        return dict(teacher64=original["u0"].double()+at,accepted=True)
    s["teacher64"]=s["u0"].double()+s["record"]["horizon"]
    result=evaluate_rollout(model,s,4,provider,bounded_fd=True)
    assert result["measured_steps"]==4 and not result["missing_steps"]
    assert len(calls)==4 and calls[1].mean()>calls[0].mean()
    assert all(torch.equal(x,s["u0"]) for x in origins)
    assert result["rows"][-1]["interval_violation"]>0
    assert result["rows"][-1]["prediction_mean"]>1
    assert result["rows"][-1]["mean_balance_residual"]>.9


def test_missing_intermediate_teachers_and_galerkin_no_maximum_principle():
    s=sample(target="galerkin_same_grid",value=1.2)
    result=evaluate_rollout(lambda u,*args:u,s,2,bounded_fd=True)
    assert result["missing_steps"]==[1]
    assert result["measured_steps"]==1
    assert all(r["interval_violation"] is None for r in result["rows"])
    assert all(not r["interval_diagnostic_applicable"] for r in result["rows"])


def test_composition_contracts_are_distinct():
    s=sample()
    geom=Geometry((5,5),(1.,1.));eq=Physics(.01,.2)
    seen=[]
    def model(u,h,*args):
        seen.append(u.clone())
        return u*math.exp(h)
    endpoint=nonlinear_endpoint_composition(model,s["u0"],.12,eq,geom,"fd_same_grid")
    assert endpoint["kind"]=="nonlinear_endpoint_composition" and endpoint["reencoded_current_state"]
    assert len(seen)==3 and not torch.equal(seen[1],seen[2])
    action=lambda z,start,stop:z*math.exp(stop-start)
    tangent=same_path_tangent_composition(action,s["u0"].double(),0,.06,.12,geom,"fd_same_grid",path_identity="one-path")
    assert tangent["rms"]<1e-15 and not tangent["reencoding"]
    with pytest.raises(ValueError,match="one declared path"):
        same_path_tangent_composition(action,s["u0"],0,.06,.12,geom,"fd_same_grid",path_identity="")


def test_independent_sensitivity_finite_difference():
    u=torch.linspace(.1,.8,25,dtype=torch.float64).reshape(1,1,5,5)
    direction=torch.ones_like(u)*.7
    result=sensitivity_panel(lambda v:v.square(),u,direction,Geometry((5,5),(1.,1.)),"fd_same_grid")
    assert all(row["rms"]<1e-10 for row in result["rows"])


@pytest.mark.parametrize("grid",[5,6])
@pytest.mark.parametrize("target",["fd_same_grid","galerkin_same_grid"])
def test_resolution_dependent_product_bound_and_signed_cancellation(grid,target):
    from tdn.analysis.response import numerics as n
    torch.manual_seed(12)
    geom=Geometry((grid,grid),(1.,1.))
    a=torch.randn(1,1,grid,grid,dtype=torch.float64)
    b=torch.randn_like(a)
    norm=lambda u:physical_metrics(u,geom,target)["rms"]
    assert norm(n.product(a,b,geom,target))<=finite_product_constant(geom,target)*norm(a)*norm(b)
    decomposition=signed_error_decomposition({"response":a,"nonlinear":-a},geom,target)
    assert decomposition["total_mse"]<1e-20
    assert decomposition["signed_cross_terms"][0]["signed_twice_inner"]<0
    assert decomposition["reconstructed_mse"]==pytest.approx(0.,abs=1e-14)
    bound=conditional_response_bound(action_error_integral=.1,nonlinear_squared_integral=.01,reaction=1.,
                                     growth_bound=2.,geom=geom,target=target)
    assert not bound["certificate"] and bound["norm_bound"]>.1
