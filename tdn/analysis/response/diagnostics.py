"""Bounded symmetry and independent-parameter response diagnostics.

These are paired descendants of one declared parent, not additional independent
fields. A generic trained convolution is not assumed rotation equivariant.
Teacher acquisition is measured separately from prediction and never passed to
the inference callable.
"""
from pathlib import Path
import time

import torch

from tdn.runtime.metadata import write_json
from .contracts import Physics
from .numerics import inner
from .references import equation_geometry, generate_reference
from .metrics import endpoint_metrics


def _predict(method, sample, record, u=None):
    from .evaluation import inference_input
    try:
        with torch.no_grad():
            answer=method(inference_input(dict(u0=sample["u0"] if u is None else u,record=record)))
        return answer,None
    except (RuntimeError,ValueError,FloatingPointError) as exc:
        return None,f"{type(exc).__name__}: {exc}"


def run(ctx, samples, methods):
    if not samples:
        raise ValueError("Response diagnostics require a declared held-out parent")
    sample=samples[0];r=sample["record"];u=sample["u0"];eq,geom=equation_geometry(r)
    rows=[];symmetry=[]
    compatible=(u.ndim==4 and u.shape[-1]==u.shape[-2] and geom.lengths[0]==geom.lengths[1])
    if compatible:
        transforms={"rotation90":lambda x:torch.rot90(x,1,(-2,-1)),"reflection_x":lambda x:torch.flip(x,(-1,))}
        for name,method in methods.items():
            ctx.budget.check();base,error=_predict(method,sample,r)
            for transform,apply in transforms.items():
                pred,failure=_predict(method,sample,r,apply(u))
                metric=endpoint_metrics(pred,apply(sample["target"]),record=r)
                defect=(pred.double()-apply(base).double()) if pred is not None and base is not None else None
                rms=float(inner(defect,defect,geom,r["target_kind"]).clamp_min(0).sqrt().sum()) if defect is not None else None
                symmetry.append(dict(metric,method=method.metadata.get("method_id",name),family=method.metadata.get("family",name),
                    transform=transform,equivariance_rms=rms,failure=failure or error,
                    scope="Same physical parent and symmetry-transformed accepted teacher; generic learned equivariance is measured, not assumed"))
    else:
        symmetry.append(dict(parent_id=r["parent_id"],status="NOT_APPLICABLE",reason="Square 2D isotropic box required for this symmetry panel"))
    # Native GPU allocations consume the separately prepared CPU intervention
    # bank. Do not regenerate the same parameter teachers in every GPU shard.
    native=torch.device(getattr(ctx,"device","cpu")).type=="cuda"
    if native:
        rows.append(dict(parent_id=r["parent_id"],target_kind=r["target_kind"],status="NOT_MEASURED",
            reason="Native parameter shifts use the CPU-prepared physical_interventions panel; local finite-difference derivative panel is not regenerated in GPU allocations"))
    for parameter in (() if native else ("kappa","reaction_rate")):
        value=float(r[parameter]);references={};teacher_seconds=0.
        if value<=0:
            rows.append(dict(parent_id=r["parent_id"],target_kind=r["target_kind"],parameter=parameter,status="NOT_APPLICABLE",
                reason="Centered parameter perturbation would leave the nonnegative physical parameter domain"));continue
        for fraction in (.01,.005):
            ctx.budget.check();delta=value*fraction;truth=[];metas=[];modified=[]
            for sign in (-1,1):
                record=dict(r,**{parameter:value+sign*delta});modified.append(record)
                physical,geometry=equation_geometry(record);tick=time.perf_counter()
                target,meta=generate_reference(u,r["horizon"],physical,geometry,r["target_kind"],ctx.protocol.get("reference"),ctx.budget)
                teacher_seconds+=time.perf_counter()-tick;truth.append(target.to(u.device));metas.append(meta)
            tangent=(truth[1]-truth[0])/(2*delta);references[fraction]=tangent
            for name,method in methods.items():
                predictions=[_predict(method,sample,record) for record in modified]
                finite=all(p is not None and bool(torch.isfinite(p).all()) for p,_ in predictions)
                error=(predictions[1][0].double()-predictions[0][0].double())/(2*delta)-tangent if finite else None
                rms=float(inner(error,error,geom,r["target_kind"]).clamp_min(0).sqrt().sum()) if finite else None
                uncertainties=[m.get("uncertainty_rms") for m in metas]
                rows.append(dict(parent_id=r["parent_id"],target_kind=r["target_kind"],grid=r["grid"],horizon=r["horizon"],
                    method=method.metadata.get("method_id",name),family=method.metadata.get("family",name),parameter=parameter,
                    parameter_value=value,relative_perturbation=fraction,epsilon=delta,tangent_error=rms,
                    reference_accepted=all(m.get("accepted",False) for m in metas),
                    reference_uncertainty_amplification=sum(uncertainties)/(2*delta) if all(v is not None for v in uncertainties) else None,
                    status="COMPLETED" if finite else "FAILED",failure=next((e for _,e in predictions if e),None),
                    scope="Independent parameter variation at fixed horizon; finite-difference derivatives conditional on this parent"))
        difference=references[.01]-references[.005]
        change=float(inner(difference,difference,geom,r["target_kind"]).clamp_min(0).sqrt().sum())
        for row in rows:
            if row.get("parameter")==parameter:
                row.update(teacher_tangent_halving_change_rms=change,teacher_acquisition_seconds=teacher_seconds,
                    teacher_cost_scope="Shared once per parent/parameter; repeated row metadata must not be summed")
    for filename,content in (("symmetry_metrics.json",symmetry),("parameter_metrics.json",rows)):
        write_json(Path(ctx.path)/filename,dict(rows=content,independent_parents=1,selection="first declared query; no outcome-informed selection"))
    return dict(symmetry_rows=len(symmetry),parameter_rows=len(rows))
