import json

import pytest
import torch

from tdn.analysis.tangent_response.performance import (
    accuracy_qualified, amortization, benchmark_cache, benchmark_calls,
    classical_controls, evaluate_schedules, freeze_schedules,
    run_performance, utility_comparison,
)


def sample():
    u=torch.ones(1,1,5,5,dtype=torch.float32)*.2
    return dict(u0=u,teacher64=u.double(),accepted=True,
                reference=dict(accepted=True,uncertainty_rms=1e-12,uncertainty_maximum=1e-12),
                record=dict(parent_id="p",query_id="q",split="validation",target_kind="fd_same_grid",
                            grid=[5,5],lengths=[1.,1.],horizon=.01,kappa=0.,reaction_rate=0.))


def row(method,schedule,query="q",split="validation",seconds=1.,error=1e-6):
    return dict(method=method,schedule=str(schedule),query_id=query,parent_id="p",split=split,
                target_kind="fd_same_grid",finite=True,reference_accepted=True,
                conservative_rms=error,conservative_maximum=error,seconds=seconds)


def test_complete_call_scopes_randomization_and_all_requested_repetitions():
    calls=[]
    def first(u,h,eq,geom,target):
        calls.append("first")
        return u
    def second(u,h,eq,geom,target):
        calls.append("second")
        return u
    methods={"first":first,"second":second}
    a=benchmark_calls(methods,[sample()],repeats=4,warmup=1,seed=13)
    b=benchmark_calls(methods,[sample()],repeats=4,warmup=1,seed=13,scope="full_transfer")
    assert len(a["rows"])==8 and len(b["rows"])==8
    assert all(v["observations"]==4 for v in a["summaries"].values())
    assert [r["method"] for r in a["rows"]]==[r["method"] for r in b["rows"]]
    assert a["scope"]=="resident_input" and b["scope"]=="full_transfer"
    assert a["energy_joules"] is None and a["monetary_cost"] is None
    assert len(calls)==24
    json.dumps(a,allow_nan=False)


def test_cache_creation_and_complete_call_parity_are_explicit():
    class Model:
        def encode(self,u,*args):return u.clone()
        def decode(self,encoded,h):return encoded+h
        def __call__(self,u,h,*args):return self.decode(self.encode(u,*args),h)
    result=benchmark_cache(Model(),sample(),repeats=2)
    assert result["exact_equal"] and result["cache_creation_charged"]
    assert result["decode_timing"]["observations"]==2
    assert "cached decode only" in result["scope"]


def test_validation_lock_rejects_diagnostic_and_missing_queries():
    with pytest.raises(ValueError,match="validation"):
        freeze_schedules([row("m",1,split="diagnostic")],source_identity="source")
    values=[row("m",1,seconds=.1),row("m",2,seconds=2),row("m",2,query="q2",seconds=2)]
    frozen=freeze_schedules(values,tolerances=[2e-5],source_identity="source")
    assert frozen["selected"][0]["schedule"]=="2"
    assert frozen["selected"][0]["requested_queries"]==2
    assert len(frozen["sha256"])==64


def test_uncertainty_missing_or_peak_failure_cannot_qualify():
    good=row("m",1)
    assert accuracy_qualified(good,2e-5)
    assert not accuracy_qualified(dict(good,conservative_maximum=3e-5),2e-5)
    assert not accuracy_qualified(dict(good,conservative_rms=None),2e-5)
    assert not accuracy_qualified(dict(good,reference_accepted=False),2e-5)


def test_strongest_qualifying_control_and_fixed_schedules():
    validation=[row("learned",1,seconds=1),row("gl4",2,seconds=2),row("adaptive",1,seconds=.01,error=3e-5)]
    frozen=freeze_schedules(validation,tolerances=[2e-5],source_identity="source")
    diagnostic=[dict(r,split="diagnostic") for r in validation]
    result=utility_comparison(diagnostic,frozen,candidate="learned",controls=["gl4","adaptive"])
    outcome=result["outcomes"][0]
    assert outcome["passed"] and outcome["strongest_qualifying_control"]=="gl4"
    assert outcome["seconds_saved"]==1
    incomplete=diagnostic+[row("gl4",2,query="q2",split="diagnostic",seconds=2)]
    outcome=utility_comparison(incomplete,frozen,candidate="learned",controls=["gl4","adaptive"])["outcomes"][0]
    assert not outcome["passed"] and outcome["status"]=="UNRESOLVED"
    assert outcome["methods"]["learned"]["coverage"]==.5


def test_negative_learning_gate_never_executes_utility():
    calls=[]
    result=run_performance([sample()],{"learned":lambda *args:calls.append(args)}, {},
                           dict(stage="learning",status="STOP",eligible_next=False))
    assert result["status"]=="NOT_RUN" and not calls
    with pytest.raises(ValueError,match="readiness"):
        run_performance([sample()],{"learned":lambda *args:args[0]}, {},
                        dict(stage="learning",status="PASS",eligible_next=True))


def test_cpu_smoke_is_implementation_only_and_real_numerical_controls():
    controls=classical_controls(schedules=[1],adaptive_tolerances=[1e-8])
    data=evaluate_schedules([sample()],controls,repeats=1,warmup=0)
    assert all(r["rms"]<1e-7 for r in data["rows"])
    assert all(accuracy_qualified(r,2e-5) for r in data["rows"])
    result=run_performance([sample()],{"anchor":lambda u,*args:u},
                           dict(profile="cpu-smoke",repeats=1,warmup=0),dict(stage="learning",status="IMPLEMENTATION_ONLY"))
    assert result["status"]=="IMPLEMENTATION_ONLY" and not result["utility_promoted"]


def test_amortization_requires_positive_measured_savings():
    assert amortization(10,None,100)["break_even_queries"] is None
    assert amortization(10,-1,100)["break_even_queries"] is None
    assert amortization(10,.2,100)["break_even_queries"]==50
