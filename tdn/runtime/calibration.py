"""Bounded A100 parity, memory and event-timing calibration, never CPU claims."""
from __future__ import annotations
import copy
import math
import os
import time
from pathlib import Path
import torch
from tdn.config import config_hash
from tdn.runtime.metadata import software_metadata, write_json
from tdn.runtime.precision import reference_precision

def relative_error(a, b):
    return float(torch.linalg.vector_norm((a-b).double()) / torch.linalg.vector_norm(a.double()).clamp_min(1e-12))

def calibrate(config: dict, run_dir: Path, device: str = "cuda") -> dict:
    if device != "cuda" or not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise ValueError("Calibration requires an allocated Slurm GPU task; CPU validation does not substitute")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Calibration requires exactly one task-visible CUDA GPU")
    prop = torch.cuda.get_device_properties(0)
    if "A100" not in prop.name or (prop.major,prop.minor)!=(8,0) or not 35*2**30 <= prop.total_memory <= 45*2**30:
        raise ValueError("Calibration requires one full A100 40 GB-class device")
    from tdn.models import build_model
    from tdn.features.local import feature_count
    reference_precision()
    torch.manual_seed(config["seed"])
    free, total = torch.cuda.mem_get_info()
    policy=config["runtime"]
    soft = int(min(policy["soft_vram_gib"]*2**30, policy["soft_vram_fraction"]*total, policy["soft_vram_fraction"]*free))
    F = feature_count(len(config["problem"]["grid"]))
    n = min(config["model"]["chunk_size"],config["training"]["microbatch_cells"],math.prod(config["problem"]["grid"]),32768)
    model = build_model(F, config["model"], t_ref=config["problem"]["t_ref"], U_ref=config["problem"]["U_ref"]).cuda()
    # Break zero-head initialization for a nontrivial parity and gradient audit.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "amplitude" in name: parameter.add_(torch.randn_like(parameter)*0.01)
    x = torch.randn(n,F,device="cuda",requires_grad=True)
    h = torch.tensor(config["horizons"]["values"][0], device="cuda")
    output = model(x,h); grad = torch.autograd.grad(output.square().sum(),x)[0]
    bf16 = copy.deepcopy(model)
    bx = x.detach().clone().requires_grad_()
    with torch.autocast("cuda",dtype=torch.bfloat16): bo = bf16(bx,h)
    bg = torch.autograd.grad(bo.square().sum(),bx)[0]
    precision = {"bf16_output_relative_error": relative_error(output.detach(),bo.detach()),
                 "bf16_gradient_relative_error": relative_error(grad,bg), "bf16_tolerance":0.03}
    precision["bf16_passed"] = max(precision["bf16_output_relative_error"],precision["bf16_gradient_relative_error"]) < .03
    selected_mode=config["precision"]["compile_mode"]
    compile_mode="default" if selected_mode == "eager" else selected_mode
    compiled = torch.compile(copy.deepcopy(model),mode=compile_mode,fullgraph=True)
    cx = x.detach().clone().requires_grad_()
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter(); co=compiled(cx,h); cg=torch.autograd.grad(co.square().sum(),cx)[0]
    torch.cuda.synchronize()
    compile_seconds=time.perf_counter()-started
    compile_memory={"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}
    precision.update(compiled_output_relative_error=relative_error(output.detach(),co.detach()),compiled_gradient_relative_error=relative_error(grad,cg),compile_tolerance=2e-5)
    precision["compile_passed"] = max(precision["compiled_output_relative_error"],precision["compiled_gradient_relative_error"]) < 2e-5
    combined_x=x.detach().clone().requires_grad_()
    with torch.autocast("cuda",dtype=torch.bfloat16): combined=compiled(combined_x,h)
    combined_grad=torch.autograd.grad(combined.square().sum(),combined_x)[0]
    precision.update(combined_output_relative_error=relative_error(output.detach(),combined.detach()),combined_gradient_relative_error=relative_error(grad,combined_grad))
    precision["combined_passed"]=max(precision["combined_output_relative_error"],precision["combined_gradient_relative_error"]) < .03
    precision["compile_mode_tested"]=compile_mode
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True); model(x,h).square().mean().backward(); optimizer.step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    timing=[]
    for _ in range(5):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record(); optimizer.zero_grad(set_to_none=True); model(x,h).square().mean().backward(); optimizer.step(); end.record(); end.synchronize()
        timing.append(start.elapsed_time(end)/1000)
    free_after,_=torch.cuda.mem_get_info()
    memory={"initial_free_bytes":free,"total_bytes":total,"soft_budget_bytes":soft,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
            "device_used_bytes_after":total-free_after}
    from tdn.numerics import Equation,Geometry,choose_substeps,reference_step,split_step
    from tdn.features.local import extract_features
    from tdn.models.solver import corrected_step
    from tdn.train.model import NormalizedModel
    from tdn.analysis.profiling import measure
    equation=Equation(config["problem"]["kappa"],config["problem"]["reaction_rate"])
    geometry=Geometry(tuple(config["problem"]["grid"]),tuple(config["problem"]["lengths"]))
    state=torch.full((1,1,*geometry.grid),.4,device="cuda",dtype=torch.float32)
    # Nonconstant data exercise stencils and nonzero splitting defects.
    state=state+.1*torch.rand_like(state)
    runtime_model=NormalizedModel(model,[0.]*F,[1.]*F).cuda()
    if selected_mode != "eager":runtime_model.compile_kernels(selected_mode)
    teacher_count=max(config["teacher"]["base_substeps"],choose_substeps(float(h.cpu()),equation,geometry))
    def short_rollout():
        result=state
        for _ in range(2):
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=config["precision"]["network_autocast"]=="bfloat16"):
                result=corrected_step(result,h,equation,geometry,runtime_model,chunk_size=config["model"]["chunk_size"])
        return result
    with torch.inference_mode():
        features=extract_features(state,equation,geometry,t_ref=model.t_ref,U_ref=model.U_ref).reshape(-1,F)
        encoded=model.encode(features) if hasattr(model,"encode") else None
        phases={}
        for name,operation in {
            "teacher_fp64":lambda:reference_step(state.double(),h.double(),equation,geometry,teacher_count),
            "split_fp32":lambda:split_step(state,h,equation,geometry),
            "features_fp32":lambda:extract_features(state,equation,geometry,t_ref=model.t_ref,U_ref=model.U_ref),
            "encode_fp32" if encoded is not None else "generic_forward_fp32":lambda:model.encode(features) if encoded is not None else model(features,h),
            "decode_fp32" if encoded is not None else "generic_forward_repeat_fp32":lambda:model.decode(encoded,h) if encoded is not None else model(features,h),
            "short_full_domain_rollout":short_rollout,
        }.items():
            _,phases[name]=measure(operation,device="cuda",warmup=1,repeats=3)
    def full_domain_backward():
        optimizer.zero_grad(set_to_none=True)
        result=state.detach().clone().requires_grad_()
        for _ in range(config["training"]["rollout_windows"]):
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=config["precision"]["network_autocast"]=="bfloat16"):
                result=corrected_step(result,h,equation,geometry,runtime_model,chunk_size=config["model"]["chunk_size"],checkpoint_chunks=config["precision"]["checkpoint_chunks"])
        result.square().mean().backward();optimizer.step()
        return result.detach()
    _,phases["full_domain_training_backward"]=measure(full_domain_backward,device="cuda",warmup=1,repeats=3)
    phase_memory_pass=all(p["peak_reserved_bytes"] < soft and p["device_used_bytes_observed_peak"] < policy["hard_memory_fraction"]*total for p in phases.values())
    passed = precision["bf16_passed"] and precision["compile_passed"] and precision["combined_passed"] and memory["peak_reserved_bytes"] < soft and compile_memory["peak_reserved_bytes"] < soft and memory["device_used_bytes_after"] < policy["hard_memory_fraction"]*total and phase_memory_pass
    result={"scope":"newly measured A100 calibration for the selected shape; no full-solver efficiency claim", "config_hash":config_hash(config),
            "passed":passed,"precision":precision,"memory":memory,"compile_seconds":compile_seconds,"compile_first_use_memory":compile_memory,
            "phase_profiles":phases,
            "encoded_points":n,"optimizer_step_seconds":timing,"software":software_metadata()}
    write_json(run_dir/"calibration.json",result)
    if not passed: raise RuntimeError("A100 parity/memory calibration failed; see calibration.json")
    return result
