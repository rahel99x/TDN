"""Actual GPU checks for new resolution prototypes, never simulated readiness."""
import os

import pytest
import torch

from tdn.analysis.adjacent.resolution_checks import audit_field
from tdn.analysis.adjacent.resolution_exploration import DeviceCorrectionEncoding,selected_step,direct_step
from tdn.analysis.frontier.numerics import CoefficientCache
from tdn.numerics import Equation,Geometry

pytestmark=pytest.mark.gpu
CASES=[(n,track) for n in (64,128) for track in ("discrete","continuum")]


@pytest.fixture(autouse=True)
def native_cuda():
    if not torch.cuda.is_available():
        if os.environ.get("TDN_REQUIRE_ADJACENT_GPU_TESTS")=="1":
            pytest.fail("Required resolution exploration suite needs actual CUDA")
        pytest.skip("No CUDA; this is not GPU validation")
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime("cuda","adjacent-resolution-exploration-gpu-tests")
    reference_precision()
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("grid,track",CASES,ids=[f"{n}-{track}" for n,track in CASES])
def test_resolution_cuda_exploration(grid,track):
    state=audit_field(grid,dtype=torch.float32)
    eq,geo=Equation(.004,3.),Geometry((grid,grid),(1.,1.))
    with torch.no_grad():
        cpu=DeviceCorrectionEncoding(state,eq,geo,track,.04,degree=3)
        gpu=DeviceCorrectionEncoding(state.cuda(),eq,geo,track,.04,degree=3)
        assert gpu.coefficients.dtype==torch.float32 and gpu.coefficients.device.type=="cuda"
        torch.testing.assert_close(gpu.query(0.).cpu(),state,rtol=0,atol=0)
        torch.testing.assert_close(gpu.query(.017).cpu(),cpu.query(.017),rtol=3e-6,atol=8e-7)
        ours,_=selected_step(state.cuda(),.04,eq,geo,track,target=2e-5)
        expected,_=selected_step(state,.04,eq,geo,track,target=2e-5)
        torch.testing.assert_close(ours.cpu(),expected,rtol=3e-6,atol=8e-7)
        cached=direct_step(state.cuda(),.04,eq,geo,track,cache=CoefficientCache())
        direct=direct_step(state.cuda(),.04,eq,geo,track)
        torch.testing.assert_close(cached,direct,rtol=3e-6,atol=8e-7)
        assert bool(torch.isfinite(ours).all())
