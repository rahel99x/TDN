"""Paired physical-bandwidth reference banks and bounded resolution diagnostics.

Every bank target is a specified *same-grid* equation. Refinement differences
are uncertainty estimates, not certificates. Spatial refinement is a separate
measurement and never silently replaces training truth. All arrays are FP64.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from tdn.analysis.frontier import numerics
from tdn.analysis.frontier.data import _temporal_teacher
from tdn.analysis.roadmap.numerics import lowpass, quadratic_df_defect, cubic_df_defect, fourier_resample
from tdn.numerics import Equation, Geometry
from tdn.runtime.metadata import software_metadata, write_json
from .diagnostics import basis_oracle, output_compression_oracle, error_metrics, _clean
from .fields import roughness_metrics
from .models import make_model, interaction_channels, MODEL_SPECS
from .protocol import digest
from .resolution_protocol import model_config_for

BANK_SCHEMA = "tdn.adjacent-resolution-bank/v1"


def _sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _safe_file(base, relative):
    base, rel = Path(base), Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError("Unsafe reference-bank path")
    answer = base/rel
    if base.is_symlink() or any((base/Path(*rel.parts[:i])).is_symlink() for i in range(1,len(rel.parts)+1)):
        raise ValueError("Symlinks are not reference-bank artifacts")
    if not answer.is_file() or not answer.resolve().is_relative_to(base.resolve()):
        raise ValueError("Missing reference-bank artifact")
    return answer


def _array(path, **arrays):
    temporary = Path(str(path)+".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{k: np.asarray(v.detach().cpu() if torch.is_tensor(v) else v, dtype=np.float64) for k,v in arrays.items()})
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary,path)
    return _sha(path)


def resolution_parent(protocol, split, index, grid=None, *, regime=None, alpha=None):
    """Grid-independent finite Fourier parent; random power prevents translations
    of one field from being counted as independent parent draws.

    Resolution-relative stress changes its physical support explicitly and has
    a different parent identity. The primary fixed-field parents do not.
    """
    p=protocol["resolution"]
    if split not in ("train","validation","evaluation","diagnostic") or type(index) is not int or index<0:
        raise ValueError("Declared split and nonnegative parent index required")
    offset={"train":0,"validation":100000,"evaluation":200000,"diagnostic":300000}[split]
    seed=int(p["seed"])+offset+index
    rng=np.random.default_rng(seed)
    regime=regime or p["regimes"][index%len(p["regimes"])]
    alpha=float(alpha if alpha is not None else p["alpha_levels"][index%len(p["alpha_levels"])])
    relative=regime=="resolution_relative_highpair"
    if relative and grid is None:
        raise ValueError("Resolution-relative workload requires its grid")
    k=int(grid)*3//8 if relative else 24
    if regime=="low": modes=[(1,0),(0,2),(2,1)]
    elif regime in ("high_pair","resolution_relative_highpair"): modes=[(k-1,1),(k,1),(1,0)]
    elif regime=="mixed": modes=[(1,0),(0,2),(4,1),(12,-3),(23,1),(24,1)]
    elif regime=="localized": modes=[(i,j) for i,j in ((1,0),(2,1),(3,-1),(4,2),(6,-2),(8,3),(12,-4),(16,4),(20,-3),(24,2))]
    elif regime=="phase_cancellation": modes=[(23,1),(24,1),(23,-1),(24,-1),(1,2)]
    elif regime=="roughness": modes=[(1,0),(0,1),(2,1),(1,3),(4,-2),(3,6),(8,3),(6,-12),(16,4),(12,-20),(24,1),(1,24)]
    elif regime=="nearly_constant": modes=[(1,0),(0,2)]
    else: raise ValueError("Unknown resolution parent regime")
    amplitudes=rng.uniform(.65,1.35,len(modes))
    if regime=="roughness": amplitudes*=np.asarray([max(abs(a),abs(b))**(-(1-alpha)) for a,b in modes])
    if regime in ("high_pair","resolution_relative_highpair"): amplitudes[-1]*=.25
    phases=rng.uniform(-math.pi,math.pi,len(modes))
    if regime=="localized":
        center=rng.uniform(0,1,2)
        phases=np.asarray([-2*math.pi*np.dot(mode,center) for mode in modes])
    if regime=="phase_cancellation":
        # Two high-pair products oppose at mode (1,0); slight random imbalance
        # keeps exact cancellation separate from noisy phase interpretation.
        phases[1]=phases[0]+.35; phases[3]=phases[2]+.35+math.pi
    rms=(1e-8 if regime=="nearly_constant" else float(p["rms"])*float(rng.uniform(.7,1.3)))
    amplitudes*=rms/math.sqrt(float(np.sum(amplitudes**2)/2))
    mean=float(p["mean"])+float(rng.uniform(-.05,.05))
    radius=float(np.sum(np.abs(amplitudes)))
    if mean-radius<0 or mean+radius>1: raise ValueError("Parent range fails analytic L1 bound; no clipping")
    identity=f"resolution-{split}-{index:04d}"
    if relative: identity+=f"-relative-n{grid}"
    if regime!=p["regimes"][index%len(p["regimes"])]: identity+="-"+regime
    # Prespecified physical variation, paired across grids, not tuned to errors.
    physics_scale=(.75,1.,1.25)[index%3]
    parent=dict(parent_id=identity,field_cluster=f"resolution-{split}-{index:04d}",split=split,index=index,
        seed=seed,regime=regime,alpha=alpha,generator="finite_2d_fourier",modes=modes,
        amplitudes=amplitudes.tolist(),phases=phases.tolist(),mean=mean,rms=rms,
        domain=[1.,1.],kappa=float(p["kappa"])*physics_scale,reaction_rate=float(p["reaction_rate"]),
        continuous_range_bound=[mean-radius,mean+radius],bandwidth_policy="resolution_relative" if relative else "fixed_physical",
        finite_field_dimension=2,nominal_hurst=1-alpha,roughness_scope="finite smooth Fourier field; alpha is a spectral-envelope label",
        normalization="continuous Fourier RMS, no clipping or grid normalization")
    parent["parent_sha256"]=digest(parent)
    return parent


def sample_resolution_parent(parent, grid):
    if grid<4 or any(2*max(abs(i),abs(j))>=grid for i,j in parent["modes"]):
        raise ValueError("Parent modes must be strictly below each grid Nyquist")
    check=dict(parent); expected=check.pop("parent_sha256",None)
    if expected is not None and digest(check)!=expected: raise ValueError("Parent metadata hash mismatch")
    x=torch.arange(grid,dtype=torch.float64)/grid
    xx,yy=torch.meshgrid(x,x,indexing="ij")
    result=torch.full((grid,grid),float(parent["mean"]),dtype=torch.float64)
    for (i,j),a,phase in zip(parent["modes"],parent["amplitudes"],parent["phases"]):
        result+=a*torch.cos(2*math.pi*(i*xx+j*yy)+phase)
    return result[None,None]


def _physics(parent,n):
    return Equation(kappa=parent["kappa"],reaction_rate=parent["reaction_rate"]),Geometry((n,n),tuple(parent["domain"]))


def converged_reference(initial,horizon,equation,geometry,track,options,check_budget=lambda:None,*,projection_grid=None):
    """Refine until Lawson *and* independent ETDRK4 meet the same criterion.

    Lawson acceptance alone does not exhaust the declared reference work. If
    its independent crosscheck is unresolved, retry at the next finer base
    within the unchanged substep and walltime bounds. Every completed trial is
    retained, and total time includes rejected reference work.
    """
    started=time.perf_counter(); trial_options=dict(options); attempts=[]; lawson_attempts=[]
    while True:
        check_budget(); trial_started=time.perf_counter()
        truth,meta=_temporal_teacher(initial,float(horizon),equation,geometry,track,trial_options,SimpleNamespace(check=check_budget))
        difference=meta.pop("_difference"); counts=meta["refinement_substeps"]
        lawson_attempts.extend(meta["temporal_refinements"])
        comparisons=[]; finite=True
        if counts:
            for count in counts[-2:]:
                value=initial.clone(); cache=numerics.CoefficientCache()
                with torch.no_grad():
                    for j in range(count):
                        if j%8==0: check_budget()
                        value=numerics.etdrk4_step(value,horizon/count,equation,geometry,track,cache=cache)
                comparisons.append(value)
            cross=error_metrics(comparisons[-1]-truth)
            refine=error_metrics(comparisons[-1]-comparisons[-2])
            finite=bool(torch.isfinite(truth).all() and all(torch.isfinite(v).all() for v in comparisons))
            floor=float(options.get("roundoff_floor",1e-12))
            rms=max(float(meta["uncertainty_rms"]),cross["error_rms"],refine["error_rms"],floor)
            maximum=max(float(meta["uncertainty_max_bound"]),cross["error_max"],refine["error_max"],floor)
            accepted=bool(meta["accepted"] and finite and max(rms,maximum)<=options["tolerance"])
            meta.update(accepted=accepted,reference_accepted=accepted,uncertainty_rms=rms,uncertainty_max_bound=maximum,
                independent_crosscheck=dict(method="Cox-Matthews ETDRK4",counts=counts[-2:],difference_from_lawson=cross,own_refinement=refine))
            if projection_grid is not None:
                # Actual projection is required: Nyquist folding is not generally
                # an RMS contraction for the sampled real-field representation.
                variations=(difference,comparisons[-1]-truth,comparisons[-1]-comparisons[-2])
                projected=[error_metrics(fourier_resample(value,(projection_grid,projection_grid))) for value in variations]
                meta["projected_temporal_uncertainty"]=dict(grid=projection_grid,
                    uncertainty_rms=max(floor,*(r["error_rms"] for r in projected)),
                    uncertainty_max_bound=max(floor,*(r["error_max"] for r in projected)),
                    differences=dict(zip(("lawson_refinement","independent_crosscheck","etdrk4_refinement"),projected)),
                    semantics="actual projected observed differences and rounding floor; not a certificate")
        else:
            meta.update(reference_accepted=False,independent_crosscheck=None)
        attempts.append(dict(meta,trial_elapsed_seconds=time.perf_counter()-trial_started,
            requested_base_substeps=trial_options["substeps"]))
        if meta["reference_accepted"] or not counts or not finite or 2*counts[-1]>options["max_substeps"]:
            break
        # A fresh three-level trial doubles its finest grid, not the cap.
        trial_options["substeps"]=2*counts[0]
    meta.update(temporal_refinements=lawson_attempts,joint_refinements=attempts,
        joint_refinement_stop="accepted" if meta["reference_accepted"] else "nonfinite" if not finite else "declared_substep_limit",
        reference_seconds=time.perf_counter()-started,reference_target="same-grid FD/nodal" if track=="discrete" else "same-grid Fourier Galerkin",
        uncertainty_semantics="maximum of observed time refinements and independent crosscheck; not a certificate",
        reference_dtype="float64",reference_device="cpu")
    return truth,meta,difference


def _bank_identity(protocol,unit):
    software=software_metadata()
    return dict(protocol_sha256=digest(protocol),unit_sha256=digest(unit),
        source_tree_sha256=software["source_tree_sha256"],software={k:software.get(k) for k in ("python","torch","numpy","scipy")})


def _verify_entry(base,row):
    for role in ("initial","reference"):
        path=_safe_file(base,row[role+"_file"])
        if _sha(path)!=row[role+"_sha256"]: raise ValueError("Reference bank array hash mismatch")


def iterate_bank_entries(bank_dir):
    base=Path(bank_dir)
    journal=json.loads(_safe_file(base,"reference_bank.json").read_text())
    if journal.get("schema")!=BANK_SCHEMA or journal.get("status")!="COMPLETED":
        raise ValueError("A completed reference bank is required")
    rows=journal["entries"]; seen=set()
    for row in rows:
        key=(row["parent_id"],row["track"],row["grid"],row["horizon"])
        if key in seen: raise ValueError("Duplicate reference-bank identity")
        seen.add(key); _verify_entry(base,row)
    return [dict(row,bank_dir=str(base.resolve())) for row in rows]


def load_bank_entry(bank_dirs,parent_id,track,grid,horizon):
    matches=[r for directory in bank_dirs for r in iterate_bank_entries(directory)
        if (r["parent_id"],r["track"],r["grid"])==(parent_id,track,int(grid)) and math.isclose(r["horizon"],float(horizon),rel_tol=0,abs_tol=1e-12)]
    if len(matches)!=1: raise ValueError(f"Expected one matching sealed bank entry, found {len(matches)}")
    row=matches[0]; arrays=[]
    for role in ("initial","reference"):
        with np.load(_safe_file(row["bank_dir"],row[role+"_file"]),allow_pickle=False) as data:
            value=np.array(data[role],copy=True)
        if value.shape!=(1,1,int(grid),int(grid)) or value.dtype!=np.float64 or not np.isfinite(value).all():
            raise ValueError("Reference array shape, precision or finiteness mismatch")
        arrays.append(torch.from_numpy(value))
    return *arrays,row


def _reference_bank_summary(journal,protocol,unit,horizons):
    return dict(status=journal["status"],reference_count=len(journal["entries"]),
        accepted_references=sum(r["accepted"] for r in journal["entries"]),
        total_attempt_seconds=sum(a.get("elapsed_seconds",0) for a in journal["attempts"]),
        expected_references=len(unit["field_indices"])*len(horizons),track=unit["track"],grid=int(unit["grid"]),split=unit["split"],
        target=protocol["resolution"]["target_by_track"][unit["track"]],scientific_success=None)


def prepare_reference_bank(protocol,unit,path,*,check_budget=lambda:None,resume=False):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    identity=_bank_identity(protocol,unit); journal_path=path/"reference_bank.json"
    p=protocol["resolution"]; n=int(unit["grid"]); track=unit["track"]; split=unit["split"]
    horizons=p["evaluation_horizons"] if split=="evaluation" else p["training_horizons"]
    expected={}
    for index in unit["field_indices"]:
        parent=resolution_parent(protocol,split,int(index),n)
        for horizon in horizons:
            expected[digest([parent["parent_id"],track,n,float(horizon)])[:20]]=(parent,float(horizon))
    if journal_path.exists():
        journal=json.loads(journal_path.read_text())
        if not resume: raise ValueError("Preserve existing reference bank; explicit exact resume only")
        if journal.get("schema")!=BANK_SCHEMA or journal.get("identity")!=identity:
            raise ValueError("Reference-bank resume source/protocol/software differs")
        seen=set()
        for row in journal["entries"]:
            key=row["entry_id"]
            if key in seen or key not in expected:
                raise ValueError("Reference-bank inventory duplicates or differs from the frozen unit")
            parent,horizon=expected[key]
            if (digest(row["parent"]),row["parent_id"],row["track"],row["grid"],row["horizon"])!=(digest(parent),parent["parent_id"],track,n,horizon):
                raise ValueError("Reference-bank inventory differs from the frozen unit")
            seen.add(key); _verify_entry(path,row)
        if journal.get("status")=="COMPLETED":
            # The outer stage can be interrupted after the inner bank commits,
            # e.g. during its final budget check or sealing. Exact recovery
            # reuses this verified bank without generating or charging teachers
            # again; the coordinator retains the original attempt separately.
            if seen!=set(expected): raise ValueError("Completed reference-bank inventory is incomplete")
            summary=_reference_bank_summary(journal,protocol,unit,horizons)
            summary_path=path/"bank_summary.json"
            if summary_path.exists():
                if json.loads(summary_path.read_text())!=summary:
                    raise ValueError("Completed reference-bank summary differs from verified inventory")
            else:
                # A process can stop between the two atomic completion writes.
                write_json(summary_path,summary)
            return summary
    else: journal=dict(schema=BANK_SCHEMA,identity=identity,status="RUNNING",entries=[],attempts=[])
    attempt=dict(started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),completed_entries_before=len(journal["entries"]))
    journal["attempts"].append(attempt); start=time.perf_counter()
    try:
        for index in unit["field_indices"]:
            check_budget(); parent=resolution_parent(protocol,split,int(index),n); initial=sample_resolution_parent(parent,n)
            equation,geometry=_physics(parent,n)
            initial_name=f"parent-{index:04d}-initial.npz"
            if not (path/initial_name).exists(): _array(path/initial_name,initial=initial)
            else:
                with np.load(_safe_file(path,initial_name),allow_pickle=False) as saved:
                    if not np.array_equal(saved["initial"],initial.numpy()):
                        raise ValueError("Existing initial array differs from frozen parent")
            for h in horizons:
                key=digest([parent["parent_id"],track,n,float(h)])[:20]
                if any(row["entry_id"]==key for row in journal["entries"]): continue
                check_budget()
                reference,refmeta,difference=converged_reference(initial,h,equation,geometry,track,p["reference"],check_budget)
                filename=f"reference-{key}.npz"
                reference_sha=_array(path/filename,reference=reference,temporal_difference=difference)
                row=dict(parent,**refmeta,parent=parent,entry_id=key,grid=n,N=n,track=track,horizon=float(h),
                    initial_file=initial_name,initial_sha256=_sha(path/initial_name),initial_key="initial",
                    reference_file=filename,reference_sha256=reference_sha,reference_key="reference",
                    protocol_sha256=identity["protocol_sha256"],source_tree_sha256=identity["source_tree_sha256"],
                    physical_cutoff=p["physical_cutoff"],split_cutoff=p["split_cutoff"])
                journal["entries"].append(_clean(row)); write_json(journal_path,journal)
        journal["status"]="COMPLETED"
    except BaseException as error:
        journal["status"]="INTERRUPTED"; attempt["error"]=f"{type(error).__name__}: {error}"
        raise
    finally:
        attempt["elapsed_seconds"]=time.perf_counter()-start
        attempt["completed_entries_after"]=len(journal["entries"])
        write_json(journal_path,_clean(journal))
        summary=_reference_bank_summary(journal,protocol,unit,horizons)
        write_json(path/"bank_summary.json",summary)
    return summary


def run_prepare(ctx):
    result=prepare_reference_bank(ctx.protocol,ctx.unit,ctx.path,check_budget=ctx.budget.check,resume=ctx.resume)
    from .core import check
    if ctx.stage+"/bank" not in ctx.identities:
        ctx.record(ctx.stage+"/bank",["D03"],metrics=result,checks=[check("all-reference-targets-accepted",result["accepted_references"],result["expected_references"],"eq",category="math")],evidence=[dict(path="reference_bank.json",sha256=_sha(ctx.path/"reference_bank.json"))])
    return result



def _collision_assessment(records):
    """Infer a feature collision only when both individual fits are feasible.

    An empty tolerance interval identifies a scalar-family limitation, while
    an unresolved oracle identifies insufficient reference information. Neither
    demonstrates that matching features require different identifiable gains.
    """
    if len(records)!=2: raise ValueError("A collision comparison needs two records")
    intervals=[r["acceptable_gain_interval"] for r in records]
    feasible=[v is not None for v in intervals]
    resolved=all(r["oracle"].get("resolved",False) for r in records)
    matched=bool(np.allclose(records[0]["features"],records[1]["features"],rtol=1e-8,atol=1e-10))
    overlap=bool(max(v[0] for v in intervals)<=min(v[1] for v in intervals)) if all(feasible) else None
    if not all(r["reference_accepted"] for r in records): reason="Reference acceptance is unresolved"
    elif not resolved: reason="Individual scalar oracle unresolved at reference uncertainty"
    elif not all(feasible): reason="Individual scalar-family target infeasible; this is not feature insufficiency"
    elif not matched: reason="The measured feature vectors do not match; no feature collision is established"
    else: reason=None
    return dict(individual_targets_feasible=feasible,individual_oracles_resolved=resolved,
        features_match=matched,acceptable_intervals_overlap=overlap,
        response_insufficiency_demonstrated=None if reason else not overlap,
        collision_inference="NA" if reason else "DEMONSTRATED" if not overlap else "NOT_DEMONSTRATED",
        collision_inference_reason=reason or "Both reference-resolved individual fits are feasible; common-gain tolerance sets compared")

class _Diagnostics:
    def __init__(self,ctx):
        self.ctx=ctx; self.p=ctx.protocol["resolution"]; self.path=ctx.path
        self.n=int(ctx.unit["grid"]); self.track=ctx.unit["track"]; self.rows=[]
        self.started=time.perf_counter(); self.reference_records=[]; self.counter=0
        self.models={family:make_model(family,self.track,model_config_for(ctx.protocol,family)).double().eval() for family in self.p["frozen_families"]}

    def flush(self,status="RUNNING",error=None):
        write_json(self.path/"resolution_diagnostic_rows.json",_clean(self.rows))
        write_json(self.path/"diagnostic_references.json",_clean(self.reference_records))
        summary=dict(status=status,grid=self.n,track=self.track,rows=len(self.rows),
            programs_measured=sorted({r["diagnostic"] for r in self.rows}),
            elapsed_seconds=time.perf_counter()-self.started,
            references=len(self.reference_records),accepted_references=sum(r["reference_accepted"] for r in self.reference_records),
            scientific_success=None,error=error,
            scope="development diagnosis; no trained-competitor or statistical superiority claim")
        write_json(self.path/"resolution_diagnostic_summary.json",_clean(summary))
        return summary

    def add(self,program,parent,h,category,method,*,arrays=None,**values):
        self.counter+=1; key=f"{program}-n{self.n}-{self.track}-{self.counter:05d}"
        row=dict(diagnostic=program,case_id=key,parent_id=parent["parent_id"],field_cluster=parent["field_cluster"],
            regime=parent["regime"],alpha=parent["alpha"],generator=parent["generator"],grid=self.n,N=self.n,
            track=self.track,T=float(h),h=float(h),domain=parent["domain"],kappa=parent["kappa"],reaction_rate=parent["reaction_rate"],
            parent=parent,category=category,method=method,
            method_role=(MODEL_SPECS.get(method,{}).get("role") or "Analytic control"),
            reference_target=self.p["target_by_track"][self.track],**values)
        if arrays:
            name=key+".npz"; row.update(array_file=name,array_sha256=_array(self.path/name,**arrays),array_keys={k:k for k in arrays})
        self.rows.append(_clean(row)); self.flush()
        return self.rows[-1]

    def teacher(self,parent,h,n=None,*,projection_grid=None):
        n=n or self.n; u=sample_resolution_parent(parent,n); eq,geo=_physics(parent,n)
        ref,meta,diff=converged_reference(u,h,eq,geo,self.track,self.p["reference"],self.ctx.budget.check,projection_grid=projection_grid)
        record=dict(meta,parent_id=parent["parent_id"],grid=n,horizon=h,track=self.track)
        self.reference_records.append(record); self.flush()
        return u,ref,dict(parent=parent,**meta,reference_uncertainty=meta["uncertainty_rms"]),diff

    def endpoint(self,parent,u,ref,meta,h,*,program="D03",schedules=None,save=True):
        eq,geo=_physics(parent,self.n); cutoff=self.p["physical_cutoff"]
        schedules=list(schedules or self.p["endpoint_steps"])
        # Randomized paired timing; initialization/cold call, warmup and
        # deployment repeats are kept separate. FP64 diagnosis is not GPU cost.
        calls={}
        for family,model in self.models.items():
            steps=schedules+([v for v in self.p["additional_classical_steps"] if v not in schedules] if family in ("df","etdrk4") and program=="D03" else [])
            for count in steps:
                def call(model=model,count=count):
                    answer=u
                    with torch.no_grad():
                        for _ in range(count): answer=model(answer,h/count,eq,geo)
                    return answer
                calls[(family,count)]=call
        timings={key:[] for key in calls}; answers={}; cold={}
        for key,call in calls.items():
            self.ctx.budget.check()
            model=self.models[key[0]]
            if hasattr(model,"cache"): model.cache.clear()
            start=time.perf_counter(); answers[key]=call(); cold[key]=time.perf_counter()-start
            for _ in range(self.p["timing"]["warmup"]): call()
        rng=np.random.default_rng(self.p["timing"]["random_seed"]+parent["index"])
        keys=list(calls)
        for repeat in range(self.p["timing"]["repeats"]):
            for index in rng.permutation(len(keys)):
                self.ctx.budget.check(); key=keys[int(index)]; start=time.perf_counter(); calls[key](); timings[key].append(time.perf_counter()-start)
        for (family,count),answer in answers.items():
            error=answer-ref; tail=error-lowpass(error,cutoff)
            compression="none" if family in ("quad2_full","quad4_full","analytic_quad_cubic","df","etdrk4") else "output"
            self.add(program,parent,h,"endpoint",family,nodes=4 if family in ("quad4_fixed","quad4_full","analytic_quad_cubic") else 2 if family not in ("df","etdrk4") else None,
                steps=count,schedule=[h/count]*count,compression=compression,output_modes=cutoff if compression=="output" else None,
                reference_uncertainty=meta["uncertainty_rms"],reference_uncertainty_max=meta["uncertainty_max_bound"],reference_accepted=meta["reference_accepted"],
                **error_metrics(error),spectral_tail_error_rms=error_metrics(tail)["error_rms"],
                median_seconds=float(np.median(timings[(family,count)])),latency_samples_seconds=timings[(family,count)],cold_seconds=cold[(family,count)],
                timing_dtype="float64",timing_device="cpu",paired_timing=True,
                offgrid_horizon=not any(math.isclose(h,t,rel_tol=0,abs_tol=1e-12) for t in self.p["training_horizons"]),
                qualified_targets={str(t):bool(meta["reference_accepted"] and error_metrics(error)["error_rms"]+meta["uncertainty_rms"]<=t and error_metrics(error)["error_max"]+meta["uncertainty_max_bound"]<=t) for t in self.p["tolerances"]},
                qualification_scope="joint RMS and maximum error, each including its reference uncertainty at the same stated tolerance",
                arrays=dict(initial=u,reference=ref,prediction=answer,residual=error,spectrum=torch.fft.fftn(error,dim=(-2,-1),norm="forward").abs()) if save and count==1 else None)
        return answers

    def floors(self,parent,u,ref,meta,h):
        eq,geo=_physics(parent,self.n); k=self.p["physical_cutoff"]
        base=self.models["df"](u,h,eq,geo); defect=ref-base; unc=meta["uncertainty_rms"] or 1e-12
        projection,info=output_compression_oracle(base,ref,k)
        self.add("D01",parent,h,"compression_floor","output_compression_oracle",**info,**error_metrics(projection-ref),
            output_modes=k,reference_uncertainty=unc,reference_accepted=meta["reference_accepted"],arrays=dict(initial=u,reference=ref,prediction=projection,residual=projection-ref))
        cubic=cubic_df_defect(u,h,eq,geo,nodes=4,target=self.track)
        for nodes in (2,4):
            channels=interaction_channels(u,h,eq,geo,track=self.track,split_modes=self.p["split_cutoff"],nodes=nodes)
            q=channels.sum(1,keepdim=True)
            for cutoff in (k,None):
                cq=lowpass(q,cutoff) if cutoff is not None else q
                cc=lowpass(cubic,cutoff) if cutoff is not None else cubic
                ch=lowpass(channels,cutoff) if cutoff is not None else channels
                dc=cq.mean((-2,-1),keepdim=True).expand_as(cq); lo=lowpass(cq,max(1,k//2))
                bands=torch.cat((dc,lo-dc,cq-lo),1)
                specifications=[("scalar",cq,(.25,1.75)),("origin_channels",ch,(.25,1.75)),("output_bands",bands,(.5,1.5)),("quadratic_cubic",torch.cat((cq,cc),1),None)]
                for name,basis,bounds in specifications:
                    self.ctx.budget.check(); matrix=basis[0].numpy()
                    oracle=basis_oracle(matrix,defect.numpy(),bounds=bounds,uncertainty=unc)
                    prediction=base+torch.as_tensor(np.einsum("i,ijk->jk",oracle["coefficients"],matrix))[None,None]
                    self.add("D01",parent,h,"oracle",name+"_oracle",nodes=nodes,output_modes=cutoff,
                        compression="output" if cutoff is not None else "none",oracle=oracle,
                        reference_informed=True,deployable=False,reference_uncertainty=unc,reference_accepted=meta["reference_accepted"],
                        **error_metrics(prediction-ref))
                # Strong analytic cubic receives exactly the same nodes and
                # output cutoff as its quadratic contained component.
                self.add("D05",parent,h,"interaction_order",f"quadratic{nodes}_plus_cubic4",nodes=nodes,cubic_nodes=4,output_modes=cutoff,
                    compression="output" if cutoff is not None else "none",**error_metrics(base+cq+cc-ref),
                    reference_uncertainty=unc,reference_accepted=meta["reference_accepted"])
            input_compressed=lowpass(quadratic_df_defect(lowpass(u,k),h,eq,geo,nodes=nodes,target=self.track),k)
            output_only=lowpass(q,k)
            for placement,increment in (("input_and_output",input_compressed),("output_only",output_only)):
                self.add("D04",parent,h,"compression_placement",f"quadratic_{placement}",nodes=nodes,output_modes=k,
                    compression=placement,**error_metrics(base+increment-ref),reference_uncertainty=unc,
                    reference_accepted=meta["reference_accepted"],low_mode_interaction_rms=error_metrics(increment)["error_rms"],
                    arrays=dict(prediction=base+increment,reference=ref,residual=base+increment-ref))
            q16=quadratic_df_defect(u,h,eq,geo,nodes=16,target=self.track)
            q32=quadratic_df_defect(u,h,eq,geo,nodes=32,target=self.track)
            self.add("D05",parent,h,"quadrature",f"quadratic_{nodes}_node",nodes=nodes,
                quadrature_error_rms=error_metrics(q-q32)["error_rms"],quadrature_refinement_rms=error_metrics(q16-q32)["error_rms"],
                reference_nodes=32,output_modes=None,reference_uncertainty=unc,reference_accepted=meta["reference_accepted"],**error_metrics(base+q-ref))
            desired=torch.fft.fftn(defect,dim=(-2,-1),norm="forward")
            parts=torch.fft.fftn(channels,dim=(-2,-1),norm="forward")
            for mode in ((0,0),(1,0),(0,1),(2,0)):
                z=complex(desired[0,0,*mode]); coefficients=[complex(parts[0,j,*mode]) for j in range(3)]
                self.add("D04",parent,h,"signed_coefficients","input_origin_channels",nodes=nodes,mode=list(mode),
                    desired_coefficient=[z.real,z.imag],channel_coefficients=[[v.real,v.imag] for v in coefficients],
                    amplitude_resolved=abs(z)>5*unc,phase_error=None if abs(z)<=5*unc else float(np.angle(sum(coefficients)/z)),
                    reference_uncertainty=unc,reference_accepted=meta["reference_accepted"],phase_semantics="radians; NA when desired coefficient is at uncertainty")

    def spatial(self,parent,h,coarse_ref,coarse_meta):
        projected=[]; metadata=[]; sizes=[]; projected_time=[]
        for factor in self.p["reference"]["spatial_factors"]:
            size=self.n*factor
            if size>self.p["reference"]["max_spatial_grid"]: continue
            _,fine,meta,time_difference=self.teacher(parent,h,size,projection_grid=self.n)
            projected.append(fourier_resample(fine,(self.n,self.n))); metadata.append(meta); sizes.append(size)
            actual=error_metrics(fourier_resample(time_difference,(self.n,self.n)))
            propagated=meta.get("projected_temporal_uncertainty")
            projected_time.append(dict(actual_lawson_difference=actual,all_crosschecks=propagated))
        if len(projected)<2: return
        difference=error_metrics(projected[-1]-projected[-2])
        temporal_accepted=bool(coarse_meta["reference_accepted"] and all(v["reference_accepted"] for v in metadata))
        known=all(v["all_crosschecks"] is not None for v in projected_time)
        reference_unc={}
        contrast_unc={}
        for metric,suffix in (("error_rms","rms"),("error_max","max_bound")):
            terms=[max(v["actual_lawson_difference"][metric],v["all_crosschecks"]["uncertainty_"+suffix]) for v in projected_time[-2:]] if known else None
            reference_unc[suffix]=difference[metric]+sum(terms) if terms is not None else None
            coarse=coarse_meta.get("uncertainty_"+suffix)
            contrast_unc[suffix]=reference_unc[suffix]+coarse if reference_unc[suffix] is not None and coarse is not None else None
        spatial_accepted=bool(known and all(v is not None and math.isfinite(v) and v<=self.p["reference"]["tolerance"] for v in reference_unc.values()))
        contrast_accepted=bool(temporal_accepted and spatial_accepted and all(v is not None and math.isfinite(v) and v<=self.p["reference"]["tolerance"] for v in contrast_unc.values()))
        self.add("D03",parent,h,"spatial_refinement","same_grid_vs_refined",fine_grids=sizes,
            **error_metrics(coarse_ref-projected[-1]),refinement_difference_rms=difference["error_rms"],
            refinement_difference_max=difference["error_max"],temporal_reference_accepted=temporal_accepted,
            spatial_resolution_accepted=spatial_accepted,contrast_reference_accepted=contrast_accepted,reference_accepted=contrast_accepted,
            projected_temporal_differences=projected_time,coarse_temporal_uncertainty_rms=coarse_meta.get("uncertainty_rms"),
            refined_reference_uncertainty_rms=reference_unc["rms"],refined_reference_uncertainty_max=reference_unc["max_bound"],
            reference_uncertainty=contrast_unc["rms"],reference_uncertainty_max=contrast_unc["max_bound"],
            uncertainty_semantics="observed spatial difference plus both projected time/crosscheck estimates; coarse teacher uncertainty additionally charged to contrast; not a certificate",
            target_scope="same-grid equation versus projected refined equation; separate from bank supervision",
            arrays=dict(reference=projected[-1],prediction=coarse_ref,residual=coarse_ref-projected[-1]))

    def trajectories(self,parent):
        times=self.p["diagnostics"]["autonomous_horizons"]
        u=sample_resolution_parent(parent,self.n); eq,geo=_physics(parent,self.n)
        truth=[u]; metas=[]
        for h in times:
            _,reference,meta,_=self.teacher(parent,h); truth.append(reference); metas.append(meta)
        for family in ("df","etdrk4","quad2_fixed","analytic_quad_cubic"):
            predicted=u; previous=0.; states=[u]; errors=[0.]; start=time.perf_counter()
            for i,h in enumerate(times):
                self.ctx.budget.check(); predicted=self.models[family](predicted,h-previous,eq,geo)
                errors.append(error_metrics(predicted-truth[i+1])["error_rms"]); states.append(predicted); previous=h
                self.add("D06",parent,h,"rollout",family,**error_metrics(predicted-truth[i+1]),
                    reference_uncertainty=metas[i]["uncertainty_rms"],reference_accepted=metas[i]["reference_accepted"],
                    schedule=[times[0]]+[times[j]-times[j-1] for j in range(1,i+1)],
                    mean=float(predicted.mean()),physical_variance=float(numerics.product(predicted,predicted,self.track).mean()-predicted.mean()**2),
                    minimum=float(predicted.min()),maximum=float(predicted.max()),finite=bool(torch.isfinite(predicted).all()),
                    accurate=bool(metas[i]["reference_accepted"] and errors[-1]+metas[i]["uncertainty_rms"]<=self.p["tolerances"][0] and error_metrics(predicted-truth[i+1])["error_max"]+metas[i]["uncertainty_max_bound"]<=self.p["tolerances"][0]),
                    qualification_scope="joint RMS and maximum with uncertainty")
            self.add("D06",parent,times[-1],"rollout_trace",family,elapsed_seconds=time.perf_counter()-start,
                elapsed_scope="total diagnostic rollout and intermediate JSON/NPZ reporting; excludes teacher generation; not isolated solver latency",
                arrays=dict(times=np.asarray([0.]+times),error_rms=np.asarray(errors),prediction=torch.cat(states,0),reference=torch.cat(truth,0)),
                reference_accepted=all(m["reference_accepted"] for m in metas))

    def amplitude(self,parent):
        from tdn.analysis.roadmap.numerics import background_response
        from .diagnostics import _resolved_slope
        n=self.n; h=.04; u=sample_resolution_parent(parent,n); mean=float(u.mean()); v=(u-mean)/parent["rms"]
        eq,geo=_physics(parent,n); constant=torch.full_like(u,mean); df=self.models["df"]
        endpoint,jac,_=background_response(constant,h,eq.reaction_rate)
        background_defect=endpoint-df(constant,h,eq,geo)
        with torch.enable_grad(): _,derivative=torch.autograd.functional.jvp(lambda value:df(value,h,eq,geo),constant,v)
        linear=numerics.heat_step(v,h,eq,geo,self.track)*jac-derivative
        rows=[]
        for epsilon in (.02,.04,.08):
            values={}; accepted=True; uncertainty=0.
            for sign in (-1,1):
                self.ctx.budget.check(); state=constant+sign*epsilon*v
                ref,meta,_=converged_reference(state,h,eq,geo,self.track,self.p["reference"],self.ctx.budget.check)
                self.reference_records.append(dict(meta,parent_id=parent["parent_id"],variant=f"epsilon{sign*epsilon}",grid=n,horizon=h,track=self.track))
                accepted &= meta["reference_accepted"]; uncertainty=max(uncertainty,meta["uncertainty_rms"])
                values[sign]=ref-df(state,h,eq,geo)-background_defect-sign*epsilon*linear
            even=(values[1]+values[-1])/2; odd=(values[1]-values[-1])/2
            q=quadratic_df_defect(constant+epsilon*v,h,eq,geo,nodes=16,target=self.track)
            c=cubic_df_defect(constant+epsilon*v,h,eq,geo,nodes=4,target=self.track)
            rows.append(self.add("D05",parent,h,"amplitude_parity","even_odd_defect",epsilon=epsilon,
                even_defect_rms=error_metrics(even)["error_rms"],odd_defect_rms=error_metrics(odd)["error_rms"],
                even_after_quadratic_rms=error_metrics(even-q)["error_rms"],odd_after_cubic_rms=error_metrics(odd-c)["error_rms"],
                reference_uncertainty=uncertainty,reference_accepted=bool(accepted),
                background_integrator_error_rms=error_metrics(background_defect)["error_rms"],linear_integrator_error_rms=error_metrics(epsilon*linear)["error_rms"],
                parity_scope="remove exact constant and AD linear backbone defects; full output",
                arrays=dict(even=even,odd=odd,quadratic=q,cubic=c)))
        for metric in ("even_defect_rms","odd_defect_rms","even_after_quadratic_rms","odd_after_cubic_rms"):
            accepted=[r for r in rows if r["reference_accepted"]]
            self.add("D05",parent,h,"observed_order","even_odd_defect",metric=metric,
                **_resolved_slope([r["epsilon"] for r in accepted],[r[metric] for r in accepted],max((r["reference_uncertainty"] for r in accepted),default=1e-12)),
                scope="finite three-amplitude fit only when at least three values exceed uncertainty")

    def collisions(self,parent):
        from tdn.analysis.portfolio.models import physical_features
        from .diagnostics import scalar_oracle,acceptable_gain_interval
        u=sample_resolution_parent(parent,self.n); eq,geo=_physics(parent,self.n); h=self.p["horizons"][-1]
        phase=dict(parent); phase["phases"]=[p+(math.pi if j==1 else 0.) for j,p in enumerate(parent["phases"])]
        phase["parent_id"]+="-phaseflip"; phase.pop("parent_sha256"); phase["parent_sha256"]=digest(phase)
        records=[]
        for item in (parent,phase):
            state,ref,meta,_=self.teacher(item,h); base=self.models["df"](state,h,eq,geo)
            q=self.models["quad2_fixed"](state,h,eq,geo)-base; desired=ref-base
            oracle=scalar_oracle(q.numpy(),desired.numpy(),(.25,1.75),meta["uncertainty_rms"])
            target=max(1e-9,(oracle.get("error_rms",error_metrics(desired)["error_rms"]))*1.05+meta["uncertainty_rms"])
            interval=acceptable_gain_interval(q.numpy(),desired.numpy(),target)
            features=physical_features(state,torch.as_tensor(h,dtype=state.dtype),eq,geo,self.track).flatten().tolist()
            records.append(dict(parent_id=item["parent_id"],oracle=oracle,acceptable_gain_interval=interval,features=features,target_rms=target,reference_accepted=meta["reference_accepted"]))
        self.add("D02",parent,h,"feature_collision","fixed_GL2_scalar_gain",pair=records,
            feature_distance=float(np.linalg.norm(np.subtract(records[0]["features"],records[1]["features"]))),
            **_collision_assessment(records),
            scope="fixed GL2 scalar gains only; no claim about a jointly learned node response")


def run_diagnostic(ctx):
    ex=_Diagnostics(ctx); p=ctx.protocol["resolution"]
    bank_dirs=[ctx.prerequisites[name] for name in ctx.unit["bank_units"]]
    metadata=[row for directory in bank_dirs for row in iterate_bank_entries(directory)]
    selected=set(p["diagnostics"]["selected_parent_indices"])
    spatial=set(p["reference"]["spatial_indices"][:p["reference"]["spatial_case_limit"]])
    try:
        # All existing bank parents retain an inexpensive representative endpoint
        # comparison. More expensive attribution is limited to declared indices.
        for row in metadata:
            ctx.budget.check(); parent=row["parent"]; index=parent["index"]
            u,ref,meta=load_bank_entry(bank_dirs,parent["parent_id"],ex.track,ex.n,row["horizon"])
            ex.endpoint(parent,u,ref,meta,row["horizon"],program="D03" if index in selected else "D08",schedules=None if index in selected else [1],save=index in selected)
            if index in selected and row["horizon"]==p["horizons"][-1]:
                ex.floors(parent,u,ref,meta,row["horizon"])
                if index in spatial: ex.spatial(parent,row["horizon"],ref,meta)
        # Fresh diagnostic variants are explicitly development evidence. They
        # do not enter training, validation selection or fresh evaluation.
        parent=resolution_parent(ctx.protocol,"diagnostic",1,ex.n,regime="high_pair")
        for h in ([.01] if ctx.protocol["profile"]=="resolution-smoke" else [.002,.01]):
            u,ref,meta,_=ex.teacher(parent,h); ex.endpoint(parent,u,ref,meta,h,program="D08",schedules=[1],save=False); ex.floors(parent,u,ref,meta,h)
        ex.collisions(parent)
        ex.amplitude(resolution_parent(ctx.protocol,"diagnostic",2,ex.n,regime="mixed"))
        ex.trajectories(parent)
        relative=resolution_parent(ctx.protocol,"diagnostic",3,ex.n,regime="resolution_relative_highpair")
        u,ref,meta,_=ex.teacher(relative,.01); ex.endpoint(relative,u,ref,meta,.01,program="D08",schedules=[1],save=False)
        # This sweep isolates envelope changes: same modes/random stream/cluster;
        # it is not ten independent fields or an infinite roughness theorem.
        for alpha in p["alpha_levels"]:
            rough=resolution_parent(ctx.protocol,"diagnostic",5,ex.n,regime="roughness",alpha=alpha)
            rough["parent_id"]+=f"-alpha{alpha:g}"; rough.pop("parent_sha256"); rough["parent_sha256"]=digest(rough)
            u,ref,meta,_=ex.teacher(rough,.01)
            ex.endpoint(rough,u,ref,meta,.01,program="D08",schedules=[1],save=False)
            ex.add("D08",rough,.01,"roughness","field_observables",initial=roughness_metrics(u),endpoint=roughness_metrics(ref),
                reference_accepted=meta["reference_accepted"],reference_uncertainty=meta["uncertainty_rms"],statistical_unit="paired envelope variants within one field cluster")
        result=ex.flush("COMPLETED")
    except BaseException as error:
        ex.flush("INTERRUPTED",f"{type(error).__name__}: {error}"); raise
    from .core import check
    ctx.record(ctx.stage+"/diagnosis",["D01","D02","D03","D04","D05","D06","D08"],metrics=result,
        checks=[check("primary-scientific-claim",None,None,category="gap",reason="Development diagnostic; trained comparisons and precision gates are separate")],
        evidence=[dict(path="resolution_diagnostic_rows.json",sha256=_sha(ctx.path/"resolution_diagnostic_rows.json"))])
    return result
