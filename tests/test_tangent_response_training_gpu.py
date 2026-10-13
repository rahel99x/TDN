"""Native CUDA architecture, staged fitting, continuation and fresh reload cases.

A skipped case is explicitly not CUDA verification. The roster is source-bound
by the Tangent Response protocol; CPU fixtures here never certify GPU work.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tdn.analysis.response.contracts import Physics
from tdn.analysis.tangent_response.models import create_model
from tdn.analysis.tangent_response import training as tr
from test_tangent_response_models import case, config as model_config
from test_tangent_response_training import samples, config as fit_config

pytestmark = pytest.mark.gpu
TARGETS = ("fd_same_grid", "galerkin_same_grid")
FAMILIES = ("learned_response", "residual_fno")
GPU_TEST_CASES = tuple("tests/test_tangent_response_training_gpu.py::"+name+"["+parameter+"]" for name, parameters in (
    ("test_tangent_cuda_model_head_and_factor_gradients", TARGETS),
    ("test_tangent_cuda_signed_causality_nulls_and_mean", TARGETS),
    ("test_tangent_cuda_cache_identity", FAMILIES),
    ("test_tangent_cuda_bridge_and_complete_cached_parity", FAMILIES),
    ("test_tangent_cuda_exact_staged_resume", FAMILIES),
    ("test_tangent_cuda_fresh_process_selected_checkpoint_reload", FAMILIES),
) for parameter in parameters)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def actual_allocation():
    if not torch.cuda.is_available():
        if os.environ.get("TDN_REQUIRE_TANGENT_RESPONSE_GPU_TESTS") == "1":
            pytest.fail("Required Tangent Response training suite has no actual CUDA device")
        pytest.skip("No actual CUDA; not Tangent Response GPU verification")
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime("cuda", "tangent-response-training-gpu-tests"); reference_precision()


@pytest.mark.parametrize("target", TARGETS, ids=TARGETS)
def test_tangent_cuda_model_head_and_factor_gradients(target):
    u,z,eq,geom = case(target, device="cuda")
    torch.manual_seed(19); model = create_model(model_config()).cuda()
    encoded = model.encode(u,eq,geom,target)
    loss = (model.response_action(encoded,z,.01,.04)-.7*z).square().mean()
    loss.backward()
    assert model.coefficient_head.bias.grad.norm()>0
    assert model.alpha_factors.weight.grad.norm()==0
    model.zero_grad(set_to_none=True)
    with torch.no_grad(): model.coefficient_head.bias.fill_(.1)
    encoded = model.encode(u,eq,geom,target)
    model.response_action(encoded,z,.01,.04).square().mean().backward()
    assert model.alpha_factors.weight.grad.norm()>0 and model.beta_factors.weight.grad.norm()>0
    assert all(torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None)


@pytest.mark.parametrize("target", TARGETS, ids=TARGETS)
def test_tangent_cuda_signed_causality_nulls_and_mean(target):
    u,z,eq,geom = case(target, device="cuda")
    model = create_model(model_config()).cuda()
    with torch.no_grad(): model.coefficient_head.bias.fill_(.2)
    encoded = model.encode(u,eq,geom,target)
    torch.testing.assert_close(model.response_action(encoded,z,.03,.03),z.double(),atol=0,rtol=0)
    for state,physics in ((u,Physics(eq.kappa,0.)),(torch.full_like(u,.4),eq)):
        cache = model.encode(state,physics,geom,target)
        torch.testing.assert_close(model.response_action(cache,z,.01,.04),model.matched_action(cache,z,.01,.04),atol=0,rtol=0)
    cache = model.encode(u,Physics(0.,eq.reaction_rate),geom,target)
    assert (model.response_action(cache,z,.01,.04)-model.matched_action(cache,z,.01,.04)).abs().max()>0
    forward = model.response_action(encoded,z,.01,.04)-model.matched_action(encoded,z,.01,.04)
    with torch.no_grad(): model.coefficient_head.bias.mul_(-1)
    cache = model.encode(u,eq,geom,target)
    reverse = model.response_action(cache,z,.01,.04)-model.matched_action(cache,z,.01,.04)
    torch.testing.assert_close(reverse,-forward,atol=2e-12,rtol=2e-9)
    assert forward.mean().abs()>0  # Full-field factors retain mean action.


@pytest.mark.parametrize("family", FAMILIES, ids=FAMILIES)
def test_tangent_cuda_cache_identity(family):
    u,_,eq,geom = case(device="cuda")
    model = create_model({**model_config(),"family":family}).cuda()
    with torch.no_grad():
        cache = model.encode(u,eq,geom,"fd_same_grid")
        model.decode(cache,.04); u.add_(.001)
        with pytest.raises(ValueError,match="state changed"): model.decode(cache,.04)
        cache = model.encode(u,eq,geom,"fd_same_grid")
        next(model.parameters()).add_(.001)
        with pytest.raises(ValueError,match="checkpoint"): model.decode(cache,.04)


@pytest.mark.parametrize("family", FAMILIES, ids=FAMILIES)
def test_tangent_cuda_bridge_and_complete_cached_parity(family):
    u,_,eq,geom = case(device="cuda")
    model = create_model({**model_config(),"family":family}).cuda().eval()
    with torch.no_grad():
        cache = model.encode(u,eq,geom,"galerkin_same_grid")
        parts = model.prediction_parts(cache,.04)
        torch.testing.assert_close(parts["output"],model(u,.04,eq,geom,"galerkin_same_grid"),atol=0,rtol=0)
        torch.testing.assert_close(parts["output"].double()-parts["ordinary_deployed"].double(),
            parts["assembly_difference"]+parts["final_rounding"],atol=0,rtol=0)
        assert parts["output"].dtype==torch.float32
        assert parts["physical_prediction"].dtype==parts["physical_correction"].dtype==parts["final_rounding"].dtype==torch.float64
        state=u
        for _ in range(4):
            state=model(state,.01,eq,geom,"galerkin_same_grid")
            assert state.dtype==torch.float32 and torch.isfinite(state).all()
    assert parts["smooth_anchor64"].dtype==torch.float64 and parts["deployed_anchor32"].dtype==torch.float32
    assert parts["work"]["anchor_evaluations"]>0 and parts["work"]["deployed_anchor_evaluations"]==1
    current=u.detach().clone().requires_grad_()
    zero=torch.tensor(0.,device="cuda",requires_grad=True)
    initial=model(current,zero,eq,geom,"galerkin_same_grid")
    torch.testing.assert_close(initial,current,atol=0,rtol=0)
    state_gradient,time_gradient=torch.autograd.grad(initial.sum(),(current,zero))
    torch.testing.assert_close(state_gradient,torch.ones_like(current),atol=2e-5,rtol=2e-5)
    expected=tr.n.rhs(current.detach().double(),eq,geom,"galerkin_same_grid").sum().float()
    torch.testing.assert_close(time_gradient,expected,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize("family", FAMILIES, ids=FAMILIES)
def test_tangent_cuda_exact_staged_resume(tmp_path,family):
    cfg = fit_config(family); data = samples("cuda")
    full = tr.fit(deepcopy(data),cfg,device="cuda",output_path=tmp_path/"full")
    tr.fit(deepcopy(data),cfg,device="cuda",output_path=tmp_path/"first",max_chunk_updates=2)
    resumed = tr.fit(deepcopy(data),cfg,device="cuda",output_path=tmp_path/"next",resume_from=tmp_path/"first")
    a = torch.load(tmp_path/"full/journal.pt",weights_only=True,map_location="cpu")
    b = torch.load(tmp_path/"next/journal.pt",weights_only=True,map_location="cpu")
    assert all(torch.equal(a["model"][key],b["model"][key]) for key in a["model"])
    for key,state in a["optimizer"]["state"].items():
        for name,value in state.items(): assert torch.equal(value,b["optimizer"]["state"][key][name])
    assert all(torch.equal(x,y) for x,y in zip(a["rng"]["cuda"],b["rng"]["cuda"]))
    assert a["scheduler_state"]==b["scheduler_state"] and a["work"]==b["work"]
    assert full["selected_state_sha256"]==resumed["selected_state_sha256"]


@pytest.mark.parametrize("family", FAMILIES, ids=FAMILIES)
def test_tangent_cuda_fresh_process_selected_checkpoint_reload(tmp_path,family):
    cfg = fit_config(family,updates=2); data = samples("cuda")
    cfg["source_tree_sha256"] = tr.software_metadata()["source_tree_sha256"]
    result = tr.fit(data,cfg,device="cuda",output_path=tmp_path/"fit")
    model,metadata = tr.load_model(tmp_path/"fit",device="cuda")
    # Only locally generated inference inputs go to the independent process.
    sample = data[-1]; eq,geom = tr.equation_geometry(sample)
    with torch.no_grad(): expected = model(sample["u0"],sample["record"]["horizon"],eq,geom,sample["record"]["target_kind"]).cpu()
    payload = tmp_path/"input.pt"
    torch.save({"u0":sample["u0"].cpu(),"record":sample["record"]},payload)
    script = '''import json,sys,torch
from tdn.analysis.tangent_response.training import load_model,equation_geometry
from tdn.runtime.preflight import verify_runtime
verify_runtime("cuda","tangent-response-fresh-reload")
s=torch.load(sys.argv[2],weights_only=True,map_location="cpu");s["u0"]=s["u0"].cuda()
m,meta=load_model(sys.argv[1],device="cuda");eq,g=equation_geometry(s)
with torch.no_grad():y=m(s["u0"],s["record"]["horizon"],eq,g,s["record"]["target_kind"])
torch.save({"output":y.cpu(),"role":meta["checkpoint_role"]},sys.argv[3])
'''
    output = tmp_path/"reloaded.pt"
    subprocess.run([sys.executable,"-c",script,str(tmp_path/"fit"),str(payload),str(output)],check=True,timeout=120,
                   cwd=Path(__file__).resolve().parents[1],env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
    actual = torch.load(output,weights_only=True)
    torch.testing.assert_close(actual["output"],expected,atol=0,rtol=0)
    assert actual["output"].dtype==torch.float32
    assert actual["role"]=="validation-selected deployment" and metadata["update"]==result["selected_update"]
    stripped={key:sample["record"][key] for key in ("query_id","parent_id","grid","lengths","horizon","target_kind","kappa","reaction_rate")}
    torch.save({"u0":sample["u0"].cpu(),"record":stripped},payload)
    completed=subprocess.run([sys.executable,"scripts/tangent_response_infer.py","--checkpoint",str(tmp_path/"fit/selected.pt"),
        "--input",str(payload),"--device","cuda","--source-sha256",cfg["source_tree_sha256"]],check=True,capture_output=True,text=True,
        timeout=120,cwd=Path(__file__).resolve().parents[1],env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
    receipt=json.loads(completed.stdout)
    assert receipt["status"]=="COMPLETED" and receipt["source_tree_sha256"]==cfg["source_tree_sha256"]
