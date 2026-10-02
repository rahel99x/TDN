"""Bounded, measured calibration with separate CARC and desktop scopes."""
from __future__ import annotations
import copy
import math
import os
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
import torch
from tdn.config import config_hash
from tdn.runtime.metadata import software_metadata, write_json
from tdn.runtime.precision import reference_precision
from tdn.runtime.preflight import execution_mode, desktop_cuda_device

def relative_error(a, b):
    return float(torch.linalg.vector_norm((a-b).double()) / torch.linalg.vector_norm(a.double()).clamp_min(1e-12))


def _finite_result(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if is_dataclass(value):
        return all(_finite_result(getattr(value, f.name)) for f in fields(value))
    if isinstance(value, dict):
        return all(_finite_result(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_result(v) for v in value)
    return value is None or not isinstance(value, (int, float)) or math.isfinite(value)


def _calibrate_desktop(config: dict, run_dir: Path, device: str) -> dict:
    """Measure eager FP32 physics/network/backward, without optimized passes."""
    if device not in ("cpu", "cuda"):
        raise ValueError("Desktop calibration device must be cpu or cuda")
    if config["precision"]["network_autocast"] != "none" or config["precision"]["compile_mode"] != "eager":
        raise ValueError("Desktop calibration currently validates eager FP32 only")
    from tdn.models import build_model
    from tdn.features.local import extract_features, feature_count
    from tdn.numerics import Equation, Geometry, choose_substeps, reference_step, split_step
    from tdn.models.solver import corrected_step
    from tdn.train.model import NormalizedModel
    from tdn.analysis.profiling import measure

    cuda = device == "cuda"
    index = desktop_cuda_device() if cuda else None
    selected = torch.device("cuda", index) if cuda else torch.device("cpu")
    policy = config["runtime"]
    if cuda:
        free, total = torch.cuda.mem_get_info(selected)
        soft = int(min(policy["soft_vram_gib"] * 2**30,
                       policy["soft_vram_fraction"] * total,
                       policy["soft_vram_fraction"] * free))
        prop = torch.cuda.get_device_properties(selected)
        hardware = {"device_index": index, "name": prop.name,
                    "compute_capability": [prop.major, prop.minor],
                    "total_bytes": int(total), "initial_free_bytes": int(free)}
    else:
        soft = total = free = None
        hardware = {"device_index": None, "name": "CPU", "cuda_measurements": "UNRUN"}
    reference_precision()
    torch.manual_seed(config["seed"])
    F = feature_count(len(config["problem"]["grid"]))
    n = min(config["model"]["chunk_size"], config["training"]["microbatch_cells"],
            math.prod(config["problem"]["grid"]), 32768)
    model = build_model(F, config["model"], t_ref=config["problem"]["t_ref"],
                        U_ref=config["problem"]["U_ref"]).to(selected)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "amplitude" in name:
                parameter.add_(torch.randn_like(parameter) * .01)
    x = torch.randn(n, F, device=selected, requires_grad=True)
    h = torch.tensor(config["horizons"]["values"][0], device=selected)
    output = model(x, h)
    gradient, = torch.autograd.grad(output.square().sum(), x)
    finite = {"encoder_forward": _finite_result(output), "encoder_gradient": _finite_result(gradient)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])

    def optimizer_step():
        optimizer.zero_grad(set_to_none=True)
        result = model(x, h)
        result.square().mean().backward()
        optimizer.step()
        return result.detach()

    _, step_profile = measure(optimizer_step, device=selected, warmup=1, repeats=3, memory_policy=policy)
    equation = Equation(config["problem"]["kappa"], config["problem"]["reaction_rate"])
    geometry = Geometry(tuple(config["problem"]["grid"]), tuple(config["problem"]["lengths"]))
    state = .4 + .1 * torch.rand((1, 1, *geometry.grid), device=selected, dtype=torch.float32)
    runtime_model = NormalizedModel(model, [0.] * F, [1.] * F).to(selected)
    teacher_count = max(config["teacher"]["base_substeps"],
                        choose_substeps(float(h.cpu()), equation, geometry))

    def short_rollout():
        result = state
        for _ in range(2):
            result = corrected_step(result, h, equation, geometry, runtime_model,
                                    chunk_size=config["model"]["chunk_size"])
        return result

    phases = {"encoder_optimizer_step": step_profile}
    with torch.inference_mode():
        features = extract_features(state, equation, geometry, t_ref=model.t_ref,
                                    U_ref=model.U_ref).reshape(-1, F)
        encoded = model.encode(features) if hasattr(model, "encode") else None
        operations = {
            "teacher_fp64": lambda: reference_step(state.double(), h.double(), equation, geometry, teacher_count),
            "split_fp32": lambda: split_step(state, h, equation, geometry),
            "features_fp32": lambda: extract_features(state, equation, geometry, t_ref=model.t_ref, U_ref=model.U_ref),
            "encode_fp32" if encoded is not None else "generic_forward_fp32":
                lambda: model.encode(features) if encoded is not None else model(features, h),
            "decode_fp32" if encoded is not None else "generic_forward_repeat_fp32":
                lambda: model.decode(encoded, h) if encoded is not None else model(features, h),
            "short_full_domain_rollout": short_rollout,
        }
        for name, operation in operations.items():
            result, phases[name] = measure(operation, device=selected, warmup=1, repeats=3, memory_policy=policy)
            finite[name] = _finite_result(result)

    def full_domain_backward():
        optimizer.zero_grad(set_to_none=True)
        result = state.detach().clone().requires_grad_()
        for _ in range(config["training"]["rollout_windows"]):
            result = corrected_step(result, h, equation, geometry, runtime_model,
                                    chunk_size=config["model"]["chunk_size"],
                                    checkpoint_chunks=config["precision"]["checkpoint_chunks"])
        result.square().mean().backward()
        finite["full_domain_parameter_gradients"] = all(
            p.grad is None or _finite_result(p.grad) for p in model.parameters())
        optimizer.step()
        return result.detach()

    result, phases["full_domain_training_backward"] = measure(
        full_domain_backward, device=selected, warmup=1, repeats=3, memory_policy=policy)
    finite["full_domain_training_backward"] = _finite_result(result)
    if cuda:
        for phase in phases.values():
            phase["soft_budget_bytes"] = soft
            phase["soft_budget_passed"] = phase["peak_reserved_bytes"] <= soft
        memory_pass = all(p["peak_reserved_bytes"] <= soft and
                          p["device_used_bytes_observed_peak"] < policy["hard_memory_fraction"] * total and
                          p["first_use_and_warmup_memory"]["peak_reserved_bytes"] <= soft
                          for p in phases.values())
        memory = {"scope": "selected local CUDA device; includes other applications in sampled device use",
                  "initial_free_bytes": int(free), "total_bytes": int(total), "soft_budget_bytes": soft,
                  "hard_device_used_fraction": policy["hard_memory_fraction"], "passed": memory_pass,
                  "peak_reserved_bytes": max(p["peak_reserved_bytes"] for p in phases.values())}
    else:
        memory_pass = True
        memory = {"scope": "CPU process memory observations; CUDA budget checks UNRUN",
                  "cuda_status": "UNRUN", "cuda_budget_passed": None,
                  "host_process_peak_rss_bytes": max((p["host_process_peak_rss_bytes"] or 0) for p in phases.values()) or None}
    reason = "Desktop baseline validates eager FP32 only; optimized parity requires a separate supported audit"
    precision = {"mode": "eager_fp32", "eager_finite_checks": finite,
                 "eager_passed": all(finite.values()), "optimized_status": "UNRUN",
                 "bf16_status": "UNRUN", "bf16_passed": False,
                 "compile_status": "UNRUN", "compile_passed": False,
                 "combined_status": "UNRUN", "combined_passed": False,
                 "unrun_reason": reason}
    report = {"scope": "newly measured desktop eager FP32 calibration; no CARC or optimized-precision claim",
              "execution_mode": "desktop", "device": str(selected), "hardware": hardware,
              "config_hash": config_hash(config), "passed": all(finite.values()) and memory_pass,
              "optimized_eligible": False, "precision": precision, "memory": memory,
              "compile_seconds": None, "compile_first_use_memory": None,
              "phase_profiles": phases, "encoded_points": n,
              "optimizer_step_seconds": step_profile["cuda_event_seconds_raw"] if cuda else step_profile["wall_seconds_raw"],
              "software": software_metadata()}
    write_json(run_dir / "calibration.json", report)
    if not report["passed"]:
        raise RuntimeError("Desktop eager finite/memory calibration failed; see calibration.json")
    return report


def calibrate(config: dict, run_dir: Path, device: str = "cuda") -> dict:
    if execution_mode() == "desktop":
        return _calibrate_desktop(config, run_dir, device)
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
