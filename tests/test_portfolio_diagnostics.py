"""Independent numerical contrasts, rather than success by implementation mirror."""
import json
import torch
import pytest

from tdn.analysis.portfolio.diagnostics import audit,run,_state
from tdn.analysis.portfolio.protocol import build_protocol
from tdn.analysis.portfolio.core import Context
from tdn.analysis.premix.neural import _RunBudget
from tdn.analysis.roadmap.numerics import fourier_resample


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_independent_teacher_audit_and_all_models(tmp_path):
    p=build_protocol();ctx=Context(p,"audit",tmp_path,{},"cpu",_RunBudget(90,None,"cpu"))
    result=audit(ctx)
    assert result["model_cases"]==46 and result["independent_teacher_tracks"]==2
    assert not any(c["verdict"]=="BAD" for r in ctx.rows for c in r["checks"])
    assert all(r["assessment"]["verdict"]=="NA" for r in ctx.rows) # Correct math alone is no benefit.


def test_bottleneck_contrasts_separate_targets_and_missing_terms(tmp_path):
    p=build_protocol();ctx=Context(p,"diagnose",tmp_path,{},"cpu",_RunBudget(90,None,"cpu"))
    result=run(ctx);rows=json.loads((tmp_path/"diagnostic_rows.json").read_text())
    assert result["error_terms_are_nonadditive"]
    mismatch=[r for r in rows if r["component"]=="equation_mismatch"]
    assert len(mismatch)==4 and max(r["error_rms"] for r in mismatch)>1e-5
    assert all(r["error_rms"] is None for r in rows if r["component"] in ("optimization","deployment"))
    assert any(r["component"]=="spatial_reference" for r in rows)
    assert all(r["reference_accepted"] for r in rows)
    profiles=json.loads((tmp_path/"profile_rows.json").read_text())
    assert any("instrumented" in r["measurement_scope"] for r in profiles)
    assert any(r["component"]=="warm_end_to_end" for r in profiles)


def test_lift_preserves_same_field_not_changed_frequencies():
    u=_state(16,"high_pair",3123)
    torch.testing.assert_close(fourier_resample(fourier_resample(u,(64,64)),(16,16)),u,atol=2e-15,rtol=2e-15)
