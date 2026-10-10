"""Independent mathematical and declaration checks for the next campaign."""
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.advance import diagnostics as d
from tdn.analysis.advance.protocol import build_protocol, validate_protocol, FAMILIES
from tdn.analysis.advance.models import make_model
from tdn.analysis.frontier.data import field_state


def test_residual_oracle_recovers_span_and_reports_nonidentifiability():
    a=torch.tensor([1.,0.,0.],dtype=torch.float64)
    b=torch.tensor([0.,1.,0.],dtype=torch.float64)
    exact=d.residual_projection(2*a-3*b,[a,b])
    assert exact["oracle_coefficients"]==pytest.approx([2,-3])
    assert exact["oracle_error_rms"]<1e-14
    missing=d.residual_projection(torch.tensor([2.,-3.,4.]),[a,b])
    assert missing["oracle_error_rms"]==pytest.approx(4/3**.5)
    duplicate=d.residual_projection(a,[a,2*a])
    assert duplicate["rank"]==1 and duplicate["condition_number"] is None
    assert "not fitted" in duplicate["scope"]


def test_protocol_fields_are_fresh_bounded_and_grid_consistent():
    namespaces=[]
    for profile in ("smoke","development","full"):
        p=validate_protocol(build_protocol(profile))
        parents=p["parents"]
        ids={r["field_cluster"] for r in parents}
        assert len(ids)==len(parents)
        assert all(ids.isdisjoint(previous) for previous in namespaces)
        namespaces.append(ids)
        for parent in parents:
            state=field_state(parent,p["train_grid"])
            assert float(state.min())>=0 and float(state.max())<=1
            # Sampling the same continuous field at integer refinement agrees.
            refined=field_state(parent,2*p["train_grid"])
            torch.testing.assert_close(state,refined[...,::2,::2],rtol=1e-12,atol=1e-12)
        for family in FAMILIES:
            make_model(family,"discrete",{**p["model_config"],**p["model_configs"].get(family,{})})
        assert p["selection"]["batch_size"]==1
        assert p["scaling"]["grids"]==([64,128] if profile=="full" else [8,16] if profile=="smoke" else [32,64])


def test_mathematical_audit_executes_all_tracks_without_mocked_numerics(tmp_path):
    p=build_protocol("smoke");rows=[]
    ctx=SimpleNamespace(protocol=p,path=tmp_path,budget=SimpleNamespace(check=lambda:None),
        record=lambda identity,mechanisms,**row:rows.append(dict(identity=identity,**row)))
    outcome=d.audit(ctx)
    assert outcome["model_cases"]==2*len(FAMILIES)
    assert len([r for r in rows if r["identity"].startswith("gradient-flow/")])==2
    assert not [(r["identity"],c) for r in rows for c in r["checks"] if c["verdict"]=="BAD"]
    assert (tmp_path/"audit_rows.json").exists()


def test_diagnostic_stage_serializes_real_metrics_and_residuals(tmp_path):
    import json
    p=build_protocol("smoke")
    p["diagnostics"].update(regimes=["high_pair"],horizons=[.03],amplitudes=[.01,.04])
    rows=[]
    ctx=SimpleNamespace(protocol=p,path=tmp_path,budget=SimpleNamespace(check=lambda:None),
        record=lambda identity,mechanisms,**row:rows.append(dict(identity=identity,**row)))
    result=d.run(ctx)
    assert result["endpoint_observations"]==16
    endpoints=json.loads((tmp_path/"diagnostic_rows.json").read_text())["rows"]
    assert {r["track"] for r in endpoints}=={"discrete","continuum"}
    assert all(r["grid"]==8 and r["reference_accepted"] for r in endpoints)
    assert all(r["unreplicated_inference_seconds"]>0 and r["error_rms"]>=0 for r in endpoints)
    for row in json.loads((tmp_path/"residual_projections.json").read_text())["rows"]:
        assert row["oracle_error_rms"]<=row["one_basis_oracle_error_rms"]+1e-15
    assert len(json.loads((tmp_path/"amplitude_rows.json").read_text())["rows"])==4
