"""Cache validity and complete-cost semantics for the adjacent D07 diagnostic."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tdn.analysis.adjacent import temporal as t
from tdn.analysis.portfolio.diagnostics import _state
from tdn.analysis.roadmap.numerics import quadratic_df_defect
from tdn.numerics import Equation,Geometry


def test_break_even_is_strict_and_unavailable_when_query_is_not_cheaper():
    assert t.nominal_break_even(6.,3.,1.)==4
    assert t.nominal_break_even(0.,3.,1.)==1
    assert t.nominal_break_even(6.,1.,1.) is None
    assert t.nominal_break_even(6.,1.,2.) is None
    with pytest.raises(ValueError):t.nominal_break_even(-1.,3.,1.)


@pytest.mark.parametrize("track",["discrete","continuum"])
def test_encoding_preserves_zero_and_matches_physical_quadratic(track):
    u=_state(8,"high_pair",931);eq=Equation(.004,3.);geo=Geometry((8,8),(1.,1.))
    encoding=t.CorrectionEncoding(u,eq,geo,track,.12,degree=10)
    torch.testing.assert_close(encoding.query(0.),u,rtol=0,atol=0)
    for h in (.001,.034,.079,.12):
        actual=encoding.correction(h)
        expected=quadratic_df_defect(u,h,eq,geo,target=track,nodes=4)
        torch.testing.assert_close(actual,expected,rtol=5e-6,atol=2e-12)
    with pytest.raises(ValueError,match="outside"):
        encoding.query(.121)


def test_cache_identity_rejects_every_changed_contract():
    u=_state(8,"high_pair",932);eq=Equation(.004,3.);geo=Geometry((8,8),(1.,1.))
    encoding=t.CorrectionEncoding(u,eq,geo,"discrete",.1,degree=3)
    encoding.validate(u.clone(),eq,geo,"discrete",.1)
    variants=[(u+1e-9,eq,geo,"discrete",.1),
              (u,Equation(.005,3.),geo,"discrete",.1),
              (u,eq,Geometry((8,8),(2.,1.)),"discrete",.1),
              (u.float(),eq,geo,"discrete",.1),
              (u,eq,geo,"continuum",.1),(u,eq,geo,"discrete",.11)]
    for args in variants:
        with pytest.raises(ValueError,match="requires encoding refresh"):
            encoding.validate(*args)
    # Mutating the caller's state does not mutate the stored representation.
    saved=encoding.state.clone();u.add_(.02)
    torch.testing.assert_close(encoding.state,saved)


def test_dense_output_is_complete_logistic_solution_for_constant_field():
    u=torch.full((1,1,4,4),.4,dtype=torch.float64)
    eq=Equation(.004,3.);geo=Geometry((4,4),(1.,1.))
    dense=t.dense_pde(u,eq,geo,"discrete",.2)
    times=np.array([.01,.07,.2]);expected=.4*np.exp(3*times)/(1+.4*np.expm1(3*times))
    np.testing.assert_allclose(dense.sol(times),np.broadcast_to(expected,(16,3)),rtol=2e-9)


class _Budget:
    def check(self):pass


def test_temporal_real_measurements_keep_workloads_and_kernel_task_separate(tmp_path):
    recorded=[]
    ctx=SimpleNamespace(path=tmp_path,device="cpu",budget=_Budget(),
        protocol={"profile":"smoke","tracks":["discrete"],"temporal":{"query_counts":[1,3],"repeats":1}},
        record=lambda *args,**kwargs:recorded.append((args,kwargs)))
    result=t.run(ctx)
    assert result["status"]=="COMPLETED" and recorded
    costs=json.loads((tmp_path/"temporal_cost_rows.json").read_text())
    assert {r["task"] for r in costs}=={"quadratic_kernel_only","complete_finite_pde"}
    assert {r["workload"] for r in costs}=={"same_state_queries","trajectory_dense_output","parameter_queries","autonomous_rollout"}
    assert all(r["measured_complete_seconds"]>0 and r["timing"]["raw"] for r in costs)
    refreshed=[r for r in costs if r["workload"] in ("parameter_queries","autonomous_rollout")]
    assert all(r["refresh_count"]==r["query_count"] for r in refreshed)
    assert all(r["rejected"] for r in json.loads((tmp_path/"temporal_cache_checks.json").read_text()))
    assert result["energy_joules"] is None and result["monetary_cost"] is None


def test_temporal_cpu_diagnostic_cannot_claim_cuda(tmp_path):
    with pytest.raises(ValueError,match="CPU"):
        t.run(SimpleNamespace(device="cuda",path=tmp_path))
