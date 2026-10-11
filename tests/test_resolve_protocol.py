import copy

import pytest

from tdn.analysis.resolve.protocol import build_protocol,validate_protocol


@pytest.mark.parametrize("profile",["smoke","development","full"])
def test_resolve_declared_portfolio_is_bounded_and_not_confirmation(profile):
    p=validate_protocol(build_protocol(profile))
    assert p["scientific_scope"]=="DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIMS"
    assert not p["confirmation"]["enabled"]
    assert {r["split"] for r in p["parents"]}=={"train","validation"}
    assert {r["kind"] for r in p["units"].values()}=={"audit","prepare","fit","interactions","fairness","kernels","memory","report"}
    assert p["units"]["memory"]["dependencies"]==["audit"]
    assert p["units"]["report"]["dependencies"]==list(p["units"])[:-1]
    assert p["portfolio_budget"]["protected_exploration"]
    assert len(p["gpu_test_cases"])==20
    for u in p["units"].values():
        h,m,s=map(int,u["walltime"].split(":"))
        assert u["seconds"]+120<=h*3600+m*60+s<=2700
        assert u["cpus"]<=8 and u["mem_gib"]<=48


def test_resolve_parent_splits_and_seeds_are_new_and_separate():
    from tdn.analysis.advance.protocol import build_protocol as old
    seen=set()
    for profile in ("smoke","development","full"):
        p=build_protocol(profile)
        ids={r["field_seed"] for r in p["parents"]}
        assert len(ids)==len(p["parents"]) and not seen & ids
        assert not ids & {r["field_seed"] for r in old(profile)["parents"]}
        seen.update(ids)
    train=[r for r in build_protocol("full")["parents"] if r["split"]=="train"]
    assert len({r["reaction_rate"]/r["kappa"] for r in train})>3
    assert any(r["regime"]=="overlapping_triads" for r in train)


@pytest.mark.parametrize("field",["seconds","dependencies","device"])
def test_resolve_rejects_modified_protocol(field):
    p=copy.deepcopy(build_protocol("smoke"))
    p["units"]["audit"][field] = {"seconds":3000,"dependencies":["report"],"device":"cuda"}[field]
    with pytest.raises(ValueError,match="declaration changed"):
        validate_protocol(p)


def test_resolve_long_optimizer_opportunities_are_predeclared():
    p=build_protocol("full");settings=p["resolve"]["fairness"]
    assert len(settings["optimizer_settings"])==3 and settings["updates"]==800
    assert settings["equal_time_seconds"]>0 and settings["tiny_updates"]>=200
    assert len([u for u in p["units"].values() if u["kind"]=="fairness"])==6
    assert p["resolve"]["kernels"]["grids"]==[64,128]
