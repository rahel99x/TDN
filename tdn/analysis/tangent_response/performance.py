"""Complete-call timing and validation-frozen matched-accuracy controls.

Cached decoding, transfers and fresh processes are distinct measurement scopes.
Native utility is gated by the scientific learning decision and source-bound
readiness/checkpoint identities. Latency alone supplies neither energy nor money.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time

import torch

from .evaluation import endpoint_metrics, physical_metrics, reference_status, sample_context
from tdn.analysis.response import numerics as n

REQUIRED_INFERENCE_COMPONENTS = ("anchor", "encoder", "path_residual_generation", "contractions",
                                 "quadrature", "transforms", "nonlinear_iterations", "output_assembly",
                                 "fp64_bridge_both_anchors_if_used")
TOLERANCES = (1e-4, 2e-5, 1e-5)


def _sync(device):
    if torch.device(device).type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA timing has no visible CUDA device; no CPU fallback")
        torch.cuda.synchronize(device)


def _call(method, sample, u0=None):
    _, eq, geom, target = sample_context(sample)
    if isinstance(method,dict) and method.get("_budget") is not None:
        method["_budget"].check()
    function = method.get("call") if isinstance(method,dict) else method
    if not callable(function):
        raise ValueError("Method must provide a complete callable")
    return function(sample["u0"] if u0 is None else u0,float(sample["record"]["horizon"]),eq,geom,target)


def _summary(values):
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return dict(observations=len(finite), median_seconds=statistics.median(finite) if finite else None,
                minimum_seconds=min(finite) if finite else None, maximum_seconds=max(finite) if finite else None,
                p99_seconds=None, p99_status="UNAVAILABLE: small repeated panel is not a p99 estimate")


def _parent_cost(rows):
    grouped=defaultdict(list)
    for row in rows:
        grouped[row["parent_id"]].append(row["seconds"])
    return sum(sum(v)/len(v) for v in grouped.values())/len(grouped) if grouped else None


def benchmark_calls(methods, samples, *, repeats=5, warmup=1, seed=0, device=None, scope="resident_input"):
    if repeats<1 or warmup<0 or scope not in ("resident_input","full_transfer"):
        raise ValueError("Positive repeats, nonnegative warmup and declared timing scope required")
    samples = list(samples)
    if not samples:
        return dict(rows=[], summaries={}, requested_queries=0, scope=scope)
    device = torch.device(device or samples[0]["u0"].device)
    _sync(device)
    preparation = time.perf_counter()
    prepared = [dict(s,u0=s["u0"].to(device)) for s in samples]
    preparation_seconds = time.perf_counter()-preparation
    host_inputs = [s["u0"].detach().cpu() for s in samples]
    rows, first_calls, warmup_seconds = [], [], 0.
    rng = random.Random(seed)
    names = sorted(methods)
    with torch.no_grad():
        for name in names:
            for index,sample in enumerate(prepared):
                try:
                    _sync(device); start = time.perf_counter()
                    _call(methods[name],sample)
                    _sync(device)
                    first_calls.append(dict(method=name,query_id=sample["record"]["query_id"],seconds=time.perf_counter()-start))
                    tick = time.perf_counter()
                    for _ in range(warmup):
                        _call(methods[name],sample)
                    _sync(device); warmup_seconds += time.perf_counter()-tick
                except (RuntimeError,ValueError,FloatingPointError,OverflowError,TimeoutError,InterruptedError) as exc:
                    first_calls.append(dict(method=name,query_id=sample["record"]["query_id"],seconds=None,
                                            status="FAILED",reason=str(exc)))
        for round_index in range(repeats):
            order = [(name,index) for name in names for index in range(len(prepared))]
            rng.shuffle(order)
            for order_index,(name,index) in enumerate(order):
                sample = prepared[index]
                row = dict(method=name,parent_id=sample["record"]["parent_id"],query_id=sample["record"]["query_id"],
                           split=sample["record"]["split"],target_kind=sample["record"]["target_kind"],
                           grid=sample["record"]["grid"],horizon=sample["record"]["horizon"],
                           round=round_index,order=order_index,scope=scope,device=str(device),batch_size=int(sample["u0"].shape[0]))
                try:
                    _sync(device); start = time.perf_counter()
                    if scope == "full_transfer":
                        metadata=methods[name] if isinstance(methods[name],dict) else {}
                        # A declared CPU adaptive solver can consume host input
                        # directly. Do not penalize it with an unnecessary
                        # host->GPU->host round trip in a host-input workload.
                        incoming = host_inputs[index] if metadata.get("execution_device")=="cpu" else host_inputs[index].to(device,copy=True)
                        output = _call(methods[name],sample,incoming)
                        output = output.to("cpu",copy=True)
                    else:
                        output = _call(methods[name],sample)
                    _sync(device); row["seconds"] = time.perf_counter()-start
                    row["finite"] = bool(torch.isfinite(output).all())
                    row["status"] = "MEASURED" if row["finite"] else "FAILED_NONFINITE"
                    function=methods[name].get("call") if isinstance(methods[name],dict) else methods[name]
                    row["work"]=dict(getattr(function,"last_work",{}))
                except (RuntimeError,ValueError,FloatingPointError,OverflowError,TimeoutError,InterruptedError) as exc:
                    row.update(status="TIME_CENSORED" if isinstance(exc,(TimeoutError,InterruptedError)) else "FAILED",seconds=None,finite=False,reason=str(exc))
                rows.append(row)
    summaries = {}
    for name in names:
        observed = [r["seconds"] for r in rows if r["method"]==name and r["status"]=="MEASURED"]
        summaries[name] = dict(_summary(observed),requested_observations=repeats*len(prepared),
                               failures=repeats*len(prepared)-len(observed))
    return dict(rows=rows,summaries=summaries,first_calls=first_calls,warmup_seconds=warmup_seconds,
                input_preparation_seconds=preparation_seconds,requested_queries=len(samples),scope=scope,
                randomization_seed=seed,synchronization="CUDA before and after each timed complete call" if device.type=="cuda" else "CPU wall clock",
                charged_components=list(REQUIRED_INFERENCE_COMPONENTS),
                scope_limits="latency encloses complete callable; no supplied teacher/path/action arrays; caller records component work identities",
                energy_joules=None,monetary_cost=None)


def benchmark_cache(model,sample,*,repeats=5,device=None):
    """Creation + decode + parity; cached decode is explicitly not full latency."""
    if repeats<1 or not callable(getattr(model,"encode",None)) or not callable(getattr(model,"decode",None)):
        raise ValueError("Cache benchmark requires explicit reusable encode/decode")
    _,eq,geom,target = sample_context(sample)
    u = sample["u0"]
    device = torch.device(device or u.device)
    if u.device != device:
        raise ValueError("Resident cache benchmark requires input on declared device")
    h = float(sample["record"]["horizon"])
    with torch.no_grad():
        _sync(device); start=time.perf_counter()
        encoded=model.encode(u,eq,geom,target)
        _sync(device); creation=time.perf_counter()-start
        full=model(u,h,eq,geom,target)
        cached=model.decode(encoded,h)
        parity=physical_metrics(cached.double()-full.double(),geom,target)
        durations=[]
        for _ in range(repeats):
            _sync(device); start=time.perf_counter()
            model.decode(encoded,h)
            _sync(device); durations.append(time.perf_counter()-start)
    return dict(cache_creation_seconds=creation,decode_timing=_summary(durations),parity=parity,
                exact_equal=bool(torch.equal(cached,full)),cache_creation_charged=True,
                cache_identity="identical current state/physics/geometry/checkpoint/device/dtype; invalidation enforced by model",
                scope="same-state cached decode only; every autonomous macrostep must re-encode",
                creation_and_median_decode_seconds=creation+statistics.median(durations))


def benchmark_fresh_process(command,*,cwd,repeats=1,timeout=60,identity,budget=None):
    """Explicit trusted argv, shell=False; launch/startup/transfer scope is separate."""
    if (not isinstance(command,(list,tuple)) or not command or not all(isinstance(x,str) for x in command)
            or repeats<1 or timeout<=0 or not identity):
        raise ValueError("Fresh-process timing requires explicit argv and workload identity")
    rows=[]
    for index in range(repeats):
        start=time.perf_counter()
        try:
            limit=min(60.,float(timeout))
            if budget is not None:
                budget.check()
                if hasattr(budget,"seconds") and hasattr(budget,"start"):
                    limit=min(limit,budget.seconds-(time.monotonic()-budget.start))
                if limit<=0:raise TimeoutError("Declared stage allowance exhausted")
            result=subprocess.run(command,cwd=Path(cwd),capture_output=True,text=True,timeout=limit,check=False)
            rows.append(dict(round=index,seconds=time.perf_counter()-start,returncode=result.returncode,
                             status="MEASURED" if result.returncode==0 else "FAILED",stderr=result.stderr[-512:]))
        except subprocess.TimeoutExpired:
            rows.append(dict(round=index,seconds=time.perf_counter()-start,status="TIME_CENSORED",returncode=None))
        except (TimeoutError,InterruptedError) as exc:
            rows.append(dict(round=index,seconds=time.perf_counter()-start,status="TIME_CENSORED",returncode=None,reason=str(exc)))
            break
    return dict(rows=rows,summary=_summary([r["seconds"] for r in rows if r["status"]=="MEASURED"]),
                workload_identity=identity,scope="fresh process total wall time; distinct from resident latency",
                command=list(command),energy_joules=None,monetary_cost=None)


def accuracy_qualified(row,tolerance):
    if tolerance<=0 or not math.isfinite(tolerance):
        raise ValueError("Physical tolerance must be positive and finite")
    rms=row.get("conservative_rms")
    maximum=row.get("conservative_maximum")
    return bool(row.get("finite",False) and row.get("reference_accepted",False)
                and rms is not None and maximum is not None
                and rms<=tolerance and maximum<=tolerance)


def classical_controls(*,schedules=(1,2,4,8),adaptive_tolerances=(1e-8,1e-10),reaction_substeps=8):
    """Actual complete shared-GL4 and independent adaptive finite-equation calls.

    Adaptive DOP853 uses CPU FP64, including its input/output transfers inside
    the call. It is a numerical competitor, not a supplied teacher endpoint.
    Accuracy qualification on the same teacher/query remains mandatory.
    """
    methods={}
    for count in schedules:
        if type(count) is not int or count<1:
            raise ValueError("Classical schedules must be positive integer work")
        def call(u,h,eq,geom,target,count=count):
            state=u
            for _ in range(count):
                state=n.anchor(state,h/count,eq,geom,target,name="shared_gl4",reaction_substeps=reaction_substeps)
            return state
        methods[f"shared_gl4/{count}"]=dict(call=call,method="shared_gl4",schedule=str(count),
                                            scope="complete shared GL4 steps; same finite equation and final time")
    for tolerance in adaptive_tolerances:
        if not math.isfinite(tolerance) or tolerance<=0:
            raise ValueError("Adaptive solver tolerance must be positive and finite")
        def call(u,h,eq,geom,target,tolerance=tolerance):
            from .oracle import independent_solve
            result,_=independent_solve(u,h,eq,geom,target,tolerance=tolerance)
            return result.to(device=u.device,dtype=u.dtype)
        methods[f"adaptive_dop853/{tolerance:g}"]=dict(call=call,method="adaptive_dop853",schedule=f"tol{tolerance:g}",
                                                      execution_device="cpu",
                                                      scope="complete CPU FP64 adaptive solver, input/output transfers included")
    return methods


def evaluate_schedules(samples,methods,*,repeats=5,warmup=1,seed=0,device=None):
    """Preserve every requested method/query, with independent accuracy and cost."""
    samples=list(samples)
    timing=benchmark_calls(methods,samples,repeats=repeats,warmup=warmup,seed=seed,device=device)
    rows=[]
    for name,method in methods.items():
        metadata=method if isinstance(method,dict) else {}
        for sample in samples:
            record=sample["record"]
            row=dict(method=metadata.get("method",name),schedule=str(metadata.get("schedule","1")),
                     split=record["split"],parent_id=record["parent_id"],query_id=record["query_id"],
                     target_kind=record["target_kind"],reference_accepted=reference_status(sample)[0],finite=False)
            elapsed=[r["seconds"] for r in timing["rows"] if r["method"]==name and r["query_id"]==record["query_id"]
                     and r["status"]=="MEASURED"]
            row["seconds"]=statistics.median(elapsed) if len(elapsed)==repeats else None
            if reference_status(sample)[0]:
                try:
                    resident=sample["u0"].to(device) if device else sample["u0"]
                    row.update(endpoint_metrics(_call(method,sample,resident),sample),status="MEASURED")
                except (RuntimeError,ValueError,FloatingPointError,OverflowError,TimeoutError,InterruptedError) as exc:
                    row.update(status="TIME_CENSORED" if isinstance(exc,(TimeoutError,InterruptedError)) else "FAILED",reason=str(exc))
            else:
                row.update(status="UNRESOLVED_REFERENCE",reason="Requested unresolved teacher retained")
            rows.append(row)
    return dict(rows=rows,timing=timing,requested_rows=len(samples)*len(methods),
                scope="accuracy and full-call resident timing on identical finite equation, input and final time")


def freeze_schedules(validation_rows,*,tolerances=TOLERANCES,source_identity):
    """Freeze one declared schedule per method/target/tolerance using validation only.

    A missing/failed requested validation query prevents qualification. The
    fastest qualifying schedule is selected from an explicit candidate list;
    diagnostic data cannot replace missing validation measurements.
    """
    rows=list(validation_rows)
    if not source_identity or any(r.get("split")!="validation" for r in rows):
        raise ValueError("Schedule selection requires source-bound validation rows only")
    grouped=defaultdict(list)
    for row in rows:
        grouped[row["method"],row["target_kind"],str(row["schedule"])].append(row)
    selected=[]
    for method,target in sorted({(k[0],k[1]) for k in grouped}):
        candidates={k[2]:v for k,v in grouped.items() if k[:2]==(method,target)}
        requested=set.union(*(set(r["query_id"] for r in v) for v in candidates.values()))
        for tolerance in tolerances:
            qualified=[]
            for schedule,values in candidates.items():
                covered={r["query_id"] for r in values}
                if covered==requested and len(values)==len(requested) and all(accuracy_qualified(r,tolerance) for r in values):
                    seconds=[r.get("seconds") for r in values]
                    if all(v is not None and math.isfinite(v) and v>=0 for v in seconds):
                        qualified.append((_parent_cost(values),schedule))
            choice=min(qualified) if qualified else None
            selected.append(dict(method=method,target_kind=target,tolerance=float(tolerance),
                                 schedule=choice[1] if choice else None,validation_parent_mean_seconds=choice[0] if choice else None,
                                 status="FROZEN" if choice else "UNRESOLVED",requested_queries=len(requested)))
    manifest=dict(schema="tdn.tangent-response-schedules/v1",selected=selected,source_identity=source_identity,
                  selection_scope="validation only; locked before diagnostic parents are evaluated")
    manifest["sha256"]=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return manifest


def utility_comparison(rows,frozen,*,tolerance=2e-5,candidate,controls):
    """Requested-cohort coverage and strongest qualifying control, no oracle policy."""
    if not frozen.get("sha256"):
        raise ValueError("Utility requires validation-frozen schedules")
    lock={(r["method"],r["target_kind"]):r for r in frozen["selected"] if r["tolerance"]==tolerance and r["status"]=="FROZEN"}
    rows=list(rows)
    if any(r.get("split")!="diagnostic" for r in rows):
        raise ValueError("Utility evaluation uses held-out diagnostic rows")
    outcomes=[]
    for target in sorted({r["target_kind"] for r in rows}):
        requested={r["query_id"] for r in rows if r["target_kind"]==target}
        per_method={}
        for method in [candidate,*controls]:
            selected=lock.get((method,target))
            values=[r for r in rows if r["target_kind"]==target and r["method"]==method and selected
                    and str(r["schedule"])==selected["schedule"]]
            qualifying=[r for r in values if accuracy_qualified(r,tolerance) and r.get("seconds") is not None
                        and math.isfinite(r["seconds"]) and r["seconds"]>=0]
            per_method[method]=dict(requested_queries=len(requested),qualified_queries=len(qualifying),
                                   coverage=len({r["query_id"] for r in qualifying})/len(requested) if requested else None,
                                   parent_mean_seconds=_parent_cost(qualifying),
                                   qualified=(len(qualifying)==len(requested) and len(values)==len(requested)))
        valid=[(v["parent_mean_seconds"],m) for m,v in per_method.items() if m in controls and v["qualified"]]
        fastest=min(valid) if valid else None
        own=per_method[candidate]
        eligible=bool(own["qualified"] and fastest)
        saving=(fastest[0]-own["parent_mean_seconds"]) if eligible else None
        outcomes.append(dict(target_kind=target,tolerance=tolerance,methods=per_method,
                             strongest_qualifying_control=fastest[1] if fastest else None,
                             seconds_saved=saving,passed=eligible and saving>0,
                             status="PASS" if eligible and saving>0 else "STOP" if eligible else "UNRESOLVED"))
    return dict(outcomes=outcomes,frozen_sha256=frozen["sha256"],candidate=candidate,controls=list(controls),
                policy_scope="fixed validation schedules; retrospective per-query fastest-method oracle forbidden",
                energy_joules=None,monetary_cost=None)


def amortization(offline_seconds,seconds_saved,queries):
    if offline_seconds<0 or queries<0:
        raise ValueError("Cost and requested workload must be nonnegative")
    return dict(break_even_queries=math.ceil(offline_seconds/seconds_saved) if seconds_saved is not None and seconds_saved>0 else None,
                projected_net_seconds=(queries*seconds_saved-offline_seconds) if seconds_saved is not None and seconds_saved>0 else None,
                scope="explicit workload and included offline seconds only; omitted costs must remain disclosed")


def run_performance(samples,methods,config,decision):
    config=dict(config or {})
    implementation_only=config.get("profile")=="cpu-smoke"
    eligible=(decision.get("stage")=="learning" and decision.get("status")=="PASS" and decision.get("eligible_next") is True)
    if not eligible and not implementation_only:
        return dict(schema="tdn.tangent-response-performance/v1",status="NOT_RUN",reason="Learning gate C not eligible",
                    learning_decision=decision,utility_promoted=False)
    if not implementation_only:
        receipt=config.get("native_readiness",{})
        identities=config.get("checkpoint_identities",{})
        if receipt.get("verified") is not True or not receipt.get("source_tree_sha256") or not identities:
            raise ValueError("Native utility requires source-bound CUDA readiness and frozen checkpoint identities")
        if any(not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v) for v in identities.values()):
            raise ValueError("Checkpoint identities must be actual SHA256 digests")
    samples=list(samples)
    timing=benchmark_calls(methods,samples,repeats=int(config.get("repeats",5)),warmup=int(config.get("warmup",1)),
                           seed=int(config.get("seed",0)),device=config.get("device"),scope="resident_input")
    transfer=benchmark_calls(methods,samples,repeats=int(config.get("repeats",5)),warmup=int(config.get("warmup",1)),
                             seed=int(config.get("seed",0)),device=config.get("device"),scope="full_transfer") if config.get("measure_transfer",True) else None
    return dict(schema="tdn.tangent-response-performance/v1",status="IMPLEMENTATION_ONLY" if implementation_only else "MEASURED",
                resident=timing,full_transfer=transfer,fresh_process=None,
                fresh_process_status="UNAVAILABLE unless explicit trusted inference command is separately measured",
                utility_promoted=False,utility_status="UNRESOLVED until accuracy-qualified validation-frozen comparison exists",
                native_readiness=config.get("native_readiness"),checkpoint_identities=config.get("checkpoint_identities"),
                energy_joules=None,monetary_cost=None)
