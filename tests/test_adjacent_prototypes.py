"""Tests address avoided work and signed cubic structure, not hoped-for wins."""
import json
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.adjacent import prototypes as p
from tdn.analysis.portfolio.diagnostics import _state
from tdn.analysis.roadmap.numerics import quadratic_df_defect,cubic_df_defect
from tdn.numerics import Equation,Geometry


def test_selector_skips_before_expensive_cubic_function_is_called():
    u=_state(8,"high_pair",101);eq=Equation(.004,3.);geo=Geometry((8,8),(1.,1.))
    def forbidden(*args,**kwargs):raise AssertionError("Cubic was evaluated on a skipped field")
    answer,selected=p.selective_correction(u,.1,eq,geo,"discrete",threshold=100.,cubic_function=forbidden)
    assert not selected
    torch.testing.assert_close(answer,quadratic_df_defect(u,.1,eq,geo,target="discrete",nodes=4))
    answer,selected=p.selective_correction(u,.1,eq,geo,"discrete",threshold=0.)
    assert selected
    expected=quadratic_df_defect(u,.1,eq,geo,target="discrete",nodes=4)+cubic_df_defect(u,.1,eq,geo,target="discrete",nodes=4)
    torch.testing.assert_close(answer,expected)


@pytest.mark.parametrize("track",["discrete","continuum"])
def test_signed_modulation_is_cubic_with_correct_sign_and_nulls(track):
    u=_state(8,"high_pair",102);eq=Equation(.004,3.);geo=Geometry((8,8),(1.,1.))
    mean=u.mean();v=u-mean
    def response(scale):
        state=mean+scale*v
        q=quadratic_df_defect(state,.1,eq,geo,target=track,nodes=4)
        return p.signed_modulation(state,q,track)
    torch.testing.assert_close(response(2.),8*response(1.),rtol=1e-10,atol=1e-16)
    torch.testing.assert_close(response(-1.),-response(1.),rtol=1e-10,atol=1e-16)
    zero=torch.zeros_like(u)
    torch.testing.assert_close(p.signed_modulation(u,zero,track),zero,rtol=0,atol=0)
    torch.testing.assert_close(response(0.),zero,rtol=0,atol=0)
    assert float(response(1.).abs().max())>0


def test_fitted_cubic_rule_is_global_bounded_and_handles_zero_direction():
    q=torch.tensor([1.,-2.,3.],dtype=torch.float64)
    fit=p.fit_modulation([(q,2*q)])
    assert fit["coefficient"]==2.
    assert p.fit_modulation([(q,20*q)])["coefficient"]==8.
    zero=p.fit_modulation([(torch.zeros_like(q),q)])
    assert zero["status"]=="UNRESOLVED_ZERO_DIRECTION"


class _Budget:
    def check(self):pass


@pytest.mark.parametrize("prototype",["E01","E02"])
def test_real_prototype_keeps_fit_and_heldout_cost_evidence(tmp_path,prototype):
    ctx=SimpleNamespace(path=tmp_path,device="cpu",budget=_Budget(),unit={"prototype_ids":[prototype]},
        protocol={"profile":"smoke","tracks":["discrete"],
                  "prototypes":{"fit_parents":2,"heldout_parents":2,"repeats":1}},
        record=lambda *args,**kwargs:None)
    result=p.run(ctx)
    assert result["status"]=="COMPLETED" and result["prototype_ids"]==[prototype]
    fits=json.loads((tmp_path/"prototype_fits.json").read_text())
    rows=json.loads((tmp_path/"prototype_measurements.json").read_text())
    assert rows and all(r["timing"]["raw"] for r in rows)
    heldout={r["parent_id"] for r in rows}
    assert heldout.isdisjoint(fits["discrete"]["parent_ids"])
    assert all(r["reference_seconds"]>0 for r in rows)
    assert all(r["reference_is_certificate"] is False for r in rows)
    decisions=json.loads((tmp_path/"prototype_decisions.json").read_text())
    assert decisions["protected_budget_fraction"]==.2
    assert decisions["decisions"][0]["confidence_interval"] is None
    assert decisions["energy_joules"] is None


def test_prototype_cpu_measurements_reject_cuda(tmp_path):
    with pytest.raises(ValueError,match="CPU"):
        p.run(SimpleNamespace(device="cuda",path=tmp_path))


def test_real_context_maps_analytic_controls_to_registered_prototype(tmp_path):
    from tdn.analysis.adjacent.core import Context
    from tdn.analysis.adjacent.protocol import build_protocol
    protocol=build_protocol("smoke")
    protocol.update(tracks=["discrete"],prototypes={"fit_parents":1,"heldout_parents":1,"repeats":1})
    ctx=Context(protocol,"explore-E01",tmp_path,{},"cpu",_Budget())
    result=p.run(ctx)
    assert result["status"]=="COMPLETED"
    assert ctx.rows and all(row["mechanism_ids"]==["E01"] for row in ctx.rows)
    assert result["decisions"][0]["frozen_thresholds"]["rms_ratio"]==1.05


@pytest.mark.parametrize("defect",["unaccepted_reference","uncertainty_sized_error","nonfinite"])
def test_unresolved_or_failed_evidence_cannot_pass_even_with_fast_latency(defect):
    gate=dict(speedup=1.1,rms_ratio=1.05,quadratic_improvement=1.1)
    common=dict(parent_id="heldout-one",error_rms=1e-6,error_max=2e-6,
                reference_uncertainty_max=1e-10,reference_accepted=True,errors_resolved=True,
                finite_prediction=True,speedup_over_analytic=10.,cubic_computed=False)
    ours=dict(common);control={"heldout-one":dict(common)}
    quadratic={"heldout-one":dict(common,error_rms=1e-5)}
    good=p.assess_cohort("E01","discrete",[ours],control,quadratic,gate,2e-5)
    assert good["utility"]=="GOOD"
    if defect=="unaccepted_reference":ours["reference_accepted"]=False
    elif defect=="uncertainty_sized_error":
        ours.update(errors_resolved=False,error_rms=1e-12)
    else:ours["finite_prediction"]=False
    result=p.assess_cohort("E01","discrete",[ours],control,quadratic,gate,2e-5)
    assert result["utility"]==("BAD" if defect=="nonfinite" else "NA")
    assert result["decision"]==("NUMERICAL_FAILURE" if defect=="nonfinite" else "REFINE_REFERENCE")
    assert result["independent_parent_count"]==1 and result["full_denominator_required"]
    assert result["worst_rms_ratio"] is None
