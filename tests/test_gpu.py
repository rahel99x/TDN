import os
import pytest
import torch

pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.cuda.is_available(),reason="No CUDA device; GPU coverage remains unvalidated")]

def require_task():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID"), "GPU checks require an allocated task"
    assert torch.cuda.device_count()==1

def test_fp64_temporal_cpu_gpu_and_gradients():
    require_task()
    from reference.temporal_core import psi3,temporal_defect
    x=torch.cat([torch.zeros(1),torch.logspace(-12,6,221)]).double()
    assert torch.allclose(psi3(x),psi3(x.cuda()).cpu(),atol=1e-14,rtol=3e-13)
    a=torch.randn(8,1,4,dtype=torch.float64,requires_grad=True)
    r=torch.rand_like(a,requires_grad=True)
    h=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    out=temporal_defect(a,r,h);cpu=torch.autograd.grad(out.sum(),(a,r,h))
    vals=[v.detach().cuda().requires_grad_() for v in (a,r,h)]
    go=temporal_defect(*vals);gpu=torch.autograd.grad(go.sum(),vals)
    assert torch.allclose(out,go.cpu(),atol=1e-13,rtol=1e-12)
    for c,g in zip(cpu,gpu):assert torch.allclose(c,g.cpu(),atol=1e-12,rtol=1e-11)

def test_rd_split_cpu_gpu_gradients():
    require_task()
    from tdn.numerics import Geometry,Equation
    from tdn.numerics.splitting import split_step
    u=(torch.rand(1,1,8,8,dtype=torch.float64)*.6+.2).requires_grad_()
    g=Geometry((8,8),(1.,1.));eq=Equation(.01,2.)
    h=torch.tensor(.1,dtype=torch.float64)
    out=split_step(u,h,eq,g);grad=torch.autograd.grad(out.square().sum(),u)[0]
    v=u.detach().cuda().requires_grad_();go=split_step(v,h.cuda(),eq,g);gg=torch.autograd.grad(go.square().sum(),v)[0]
    assert torch.allclose(out,go.cpu(),atol=1e-12,rtol=1e-11)
    assert torch.allclose(grad,gg.cpu(),atol=1e-11,rtol=1e-10)

def test_compiled_temporal_jets_and_bf16_decoder():
    require_task()
    from reference.temporal_core import temporal_defect,temporal_jets,TemporalParameterMLP
    compiled=torch.compile(temporal_defect,fullgraph=True)
    a=torch.tensor([[[.3,-.4,.6]]],device="cuda",dtype=torch.float64)
    r=torch.tensor([[[0.,1.2,8.]]],device="cuda",dtype=torch.float64)
    for hv in [0.,.04,.5,2.]:
        h=torch.tensor(hv,device="cuda",dtype=torch.float64,requires_grad=True)
        assert torch.allclose(compiled(a,r,h),temporal_defect(a,r,h),atol=1e-12,rtol=1e-11)
        jets=temporal_jets(a,r,h)
        f=jets[0].sum()
        for derivative in jets[1:]:
            f,=torch.autograd.grad(f,h,create_graph=True)
            assert torch.allclose(f,derivative.sum(),atol=2e-11,rtol=2e-10)
    net=TemporalParameterMLP(8,1,width=32).cuda()
    with torch.no_grad():net.amplitude.weight.fill_(.01)
    x=torch.randn(64,8,device="cuda",requires_grad=True);h=torch.tensor(.2,device="cuda")
    reference=net(x,h)
    with torch.autocast("cuda",dtype=torch.bfloat16):candidate=net(x,h)
    assert candidate.dtype==torch.float32
    assert torch.linalg.vector_norm(candidate-reference)/torch.linalg.vector_norm(reference).clamp_min(1e-12) < .03

def test_allocated_gpu_usr1_resume_checkpoint(tmp_path):
    require_task()
    import json,signal,subprocess,sys,time,yaml
    from pathlib import Path
    from tdn.config import load_config
    from tdn.data import generate_dataset
    from tdn.train import load_checkpoint
    root=Path(__file__).resolve().parents[1]
    config=load_config(root/"configs/smoke.yaml")
    config["problem"]["grid"]=[4,4]
    config["data"].update(train_count=2,validation_count=1,diagnostic_count=1)
    config["training"].update(max_steps=500,checkpoint_every_steps=1,validation_every_steps=100)
    path=tmp_path/"config.yaml";path.write_text(yaml.safe_dump(config))
    data=tmp_path/"data";generate_dataset(config,data)
    run=tmp_path/"training"
    stdout=tmp_path/"child.stdout";stderr=tmp_path/"child.stderr"
    with stdout.open("w") as out,stderr.open("w") as err:
        child=subprocess.Popen([sys.executable,"-m","tdn.cli","train","--config",str(path),"--run-dir",str(run),"--dataset",str(data),"--device","cuda"],cwd=root,stdout=out,stderr=err)
        try:
            deadline=time.monotonic()+90
            while not (run/"checkpoints/latest.json").exists():
                if child.poll() is not None:pytest.fail(f"GPU child exited before signal: {stderr.read_text()}")
                if time.monotonic()>deadline:pytest.fail("GPU checkpoint startup exceeded90seconds")
                time.sleep(.02)
            child.send_signal(signal.SIGUSR1)
            assert child.wait(timeout=30)==75,stderr.read_text()
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=30)
    checkpoint=load_checkpoint(run/"checkpoints")
    assert checkpoint["status"]=="PAUSED_NEEDS_RESUME"
    assert checkpoint["global_step"]>=1
    assert checkpoint["sampler"]["committed_cursor"]==checkpoint["global_step"]
    assert json.loads((run/"stage.json").read_text())["status"]=="PAUSED_NEEDS_RESUME"
    assert not (run/"COMPLETED").exists()
    # Resume in a fresh process for two complete updates with the exact same
    # scientific config and software/thread policy. A bounded API stop does
    # not falsely complete the originally prespecified 500-step run.
    resumed=tmp_path/"resumed"
    program="""import json,sys,torch
from pathlib import Path
from tdn.config import load_config
from tdn.runtime.storage import configure_storage
from tdn.runtime.preflight import verify_runtime
from tdn.train import load_checkpoint,train
configure_storage();verify_runtime('cuda','train')
torch.set_num_threads(1);torch.set_num_interop_threads(1)
c=load_config(sys.argv[1]);p=load_checkpoint(sys.argv[3])
r=train(c,Path(sys.argv[2]),Path(sys.argv[4]),device='cuda',resume=Path(sys.argv[3]),max_steps=p['global_step']+2)
assert r['global_step']==p['global_step']+2
assert r['status']=='PAUSED_BUDGET'
print(json.dumps({'global_step':r['global_step'],'status':r['status']}))
"""
    resumed_process=subprocess.run([sys.executable,"-c",program,str(path),str(data),str(run/"checkpoints/last.pt"),str(resumed)],cwd=root,capture_output=True,text=True,timeout=90)
    assert resumed_process.returncode==0,resumed_process.stderr
    resumed_checkpoint=load_checkpoint(resumed/"checkpoints")
    assert resumed_checkpoint["global_step"]==checkpoint["global_step"]+2
    assert resumed_checkpoint["sampler"]["committed_cursor"]==checkpoint["sampler"]["committed_cursor"]+2
