"""Bounded, reference-informed adjacent-study diagnostics D01–D06 and D08.

These are development experiments, not deployment oracles and not a trained
competitor benchmark. Every endpoint has an explicit equation target. The
``continuum`` implementation means the supplied finite Galerkin equation;
separate projected fine-grid contrasts are recorded before continuum claims.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import lsq_linear
import torch

from tdn.analysis.frontier.data import lawson_reference
from tdn.analysis.frontier import numerics
from tdn.analysis.portfolio.models import physical_features
from tdn.analysis.roadmap.numerics import (background_response, cubic_df_defect, fourier_resample,
    lowpass, quadratic_df_defect)
from tdn.numerics import Equation, Geometry
from .fields import make_parent, sample_field, roughness_metrics
from .models import gain_bounds, interaction_channels, make_model

PROGRAMS = ("D01", "D02", "D03", "D04", "D05", "D06", "D08")


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    return value


def _write(path, value):
    path.write_text(json.dumps(_clean(value), indent=2, allow_nan=False) + "\n")


def error_metrics(error):
    """Unnormalized physical-state errors; no division by a tiny defect."""
    array = error.detach().cpu().numpy() if torch.is_tensor(error) else np.asarray(error)
    return dict(error_rms=float(np.sqrt(np.mean(array**2))),
                error_max=float(np.max(np.abs(array))),
                mean_error=float(np.mean(array)),
                combined_objective=float(np.mean(array**2) + .1*np.max(array**2)))


def scalar_oracle(q, defect, bounds=None, uncertainty=1e-12):
    """Exact squared-L2 scalar projection, explicitly unresolved near zero.

    Bounds apply to the effective gain, not an internal network logit. The
    combined RMS/maximum objective is *evaluated*, never called optimized.
    """
    q, defect = np.asarray(q, dtype=np.float64), np.asarray(defect, dtype=np.float64)
    if q.shape != defect.shape or not np.isfinite(q).all() or not np.isfinite(defect).all():
        raise ValueError("Finite shape-matched scalar basis and defect required")
    q_rms, d_rms = np.sqrt(np.mean(q*q)), np.sqrt(np.mean(defect*defect))
    floor = max(float(uncertainty), 64*np.finfo(float).eps)
    resolved = bool(q_rms > 5*floor and d_rms > 5*floor)
    if not resolved:
        return dict(resolved=False, coefficient=None, unrestricted_coefficient=None,
                    reason="correction or desired defect at reference/numerical floor",
                    correction_rms=float(q_rms), defect_rms=float(d_rms), bounds=bounds)
    coefficient = float(np.vdot(q, defect).real / np.vdot(q, q).real)
    fitted = coefficient if bounds is None else float(np.clip(coefficient, *bounds))
    return dict(resolved=True, coefficient=fitted, unrestricted_coefficient=coefficient,
                correction_rms=float(q_rms), defect_rms=float(d_rms), bounds=bounds,
                coefficient_uncertainty_bound=float(floor/q_rms),
                **error_metrics(fitted*q-defect))


def acceptable_gain_interval(q, defect, target_rms, bounds=(.25, 1.75)):
    """Closed set of gains satisfying an RMS target; None means empty.

    This is a tolerance set, not a confidence interval. A nearly flat objective
    can yield the whole admissible range despite apparently different optima.
    """
    q, defect = np.asarray(q, float), np.asarray(defect, float)
    aa, bb = float(np.mean(q*q)), float(np.mean(q*defect))
    cc = float(np.mean(defect*defect) - float(target_rms)**2)
    if aa <= np.finfo(float).tiny:
        return list(bounds) if cc <= 0 else None
    discriminant = bb*bb - aa*cc
    if discriminant < -64*np.finfo(float).eps*max(bb*bb, abs(aa*cc), np.finfo(float).tiny):
        return None
    radius = math.sqrt(max(0., discriminant))/aa
    center = bb/aa
    lo, hi = max(bounds[0], center-radius), min(bounds[1], center+radius)
    return [float(lo), float(hi)] if lo <= hi else None


def basis_oracle(basis, defect, *, bounds=None, uncertainty=1e-12,
                 reference_difference=None):
    """Stable SVD/bounded least squares with identifiable-space diagnostics.

    Algebraic floating-point rank is not a reference-resolved response space.
    Singular directions whose unit-coefficient RMS response is no larger than
    five times the reference uncertainty are excluded before fitting. This
    conservative diagnostic truncation prevents enormous gains on roundoff
    channels from masquerading as useful new spatial directions. It is not a
    theorem about the rank of the exact analytic quadratic form.
    """
    defect = np.asarray(defect, float).ravel()
    basis = np.asarray(basis, float).reshape(len(basis), -1).T
    if basis.shape[0] != defect.size or not np.isfinite(basis).all() or not np.isfinite(defect).all():
        raise ValueError("Finite shape-matched basis and defect required")
    if not math.isfinite(uncertainty) or uncertainty <= 0:
        raise ValueError("Positive finite reference uncertainty required")
    scale = max(float(np.sqrt(np.mean(defect**2))), float(np.max(np.abs(basis))), uncertainty)
    u, singular, vh = np.linalg.svd(basis, full_matrices=False)
    algebraic_threshold = max(basis.shape)*np.finfo(float).eps*singular[0] if singular.size else 0.
    threshold = max(algebraic_threshold,5*uncertainty*math.sqrt(defect.size))
    algebraic_rank = int(np.sum(singular > algebraic_threshold))
    rank = int(np.sum(singular > threshold))
    resolved_basis = (u[:,:rank]*singular[:rank])@vh[:rank] if rank else np.zeros_like(basis)
    rms_columns = np.sqrt(np.mean(basis**2, axis=0))
    resolved = bool(rank and np.max(rms_columns) > 5*uncertainty
                    and np.sqrt(np.mean(defect**2)) > 5*uncertainty)
    nominal = np.linalg.lstsq(basis/scale, defect/scale, rcond=None)[0]
    coefficients = (vh[:rank].T@((u[:,:rank].T@defect)/singular[:rank])
                    if rank else np.zeros(basis.shape[1]))
    unrestricted = coefficients.copy()
    optimizer = dict(kind="reference-resolved truncated SVD least squares", success=True, optimality=None)
    if bounds is not None:
        lower = np.broadcast_to(np.asarray(bounds[0], float), coefficients.shape)
        upper = np.broadcast_to(np.asarray(bounds[1], float), coefficients.shape)
        if np.any(lower >= upper):
            raise ValueError("Basis bounds require a nonempty interval per coefficient")
        result = lsq_linear(resolved_basis/scale, defect/scale, bounds=(lower, upper),
                            tol=1e-13, max_iter=300)
        coefficients = result.x
        optimizer = dict(kind="scaled bounded least squares on reference-resolved basis", success=bool(result.success),
                         optimality=float(result.optimality), iterations=result.nit)
    prediction = basis@coefficients
    perturbation = np.zeros_like(defect) if reference_difference is None else np.asarray(reference_difference, float).ravel()
    delta = (vh[:rank].T@((u[:,:rank].T@perturbation)/singular[:rank])
             if rank else np.zeros(basis.shape[1]))
    coefficient_radius = math.sqrt(defect.size)*uncertainty/singular[rank-1] if rank else None
    return dict(resolved=resolved, coefficients=coefficients.tolist(),
        unrestricted_coefficients=unrestricted.tolist(), bounds=bounds,
        effective_rank=rank, algebraic_rank=algebraic_rank, columns=basis.shape[1], singular_values=singular.tolist(),
        reference_resolved_singular_threshold=threshold,algebraic_singular_threshold=algebraic_threshold,
        singular_response_rms=(singular/math.sqrt(defect.size)).tolist(),
        numerical_direction_truncation=rank < algebraic_rank,
        nominal_algebraic_coefficients=nominal.tolist(),
        nominal_coefficients_scope="raw floating-point algebraic fit; not inferred or deployable gains, may amplify numerical noise",
        condition_number=float(singular[0]/singular[-1]) if rank == basis.shape[1] else None,
        rank_deficient=rank < basis.shape[1], column_rms=rms_columns.tolist(),
        coefficient_refinement_sensitivity_l2=float(np.linalg.norm(delta)),
        coefficient_uncertainty_radius_l2=coefficient_radius,
        sensitivity_scope="identifiable SVD subspace; unidentifiable nullspace unbounded",
        coefficient_values_identifiable=bool(resolved and rank == basis.shape[1]
            and coefficient_radius < .05*max(1., float(np.linalg.norm(coefficients)))),
        practical_identifiability_criterion="coefficient uncertainty radius below 5% of max(1, coefficient norm)",
        optimizer=optimizer, objective_optimized="squared_L2_only", **error_metrics(prediction-defect))


def output_compression_oracle(base,reference,cutoff):
    """Orthogonal L2 floor for *any* correction confined to retained modes.

    If d=reference-base and P is the same Fourier projector, then
    ||d-c||²=||(I-P)d||²+||Pd-c||² for every c=P(c). This establishes an
    L2/RMS floor, not a maximum-norm floor. Reference uncertainty still applies.
    """
    if base.shape != reference.shape or base.ndim != 4:
        raise ValueError("Shape-matched scalar field tensors required")
    desired=reference-base
    projected=lowpass(desired,cutoff)
    prediction=base+projected
    return prediction,dict(output_compression_rms_floor=error_metrics(desired-projected)["error_rms"],
        floor_scope="orthogonal RMS/L2 floor for arbitrary same-cutoff additive correction",
        maximum_norm_floor=None,
        maximum_error_semantics="maximum error of the minimum-L2 projection, not a minimax lower bound",
        projection_idempotence_max=float((lowpass(projected,cutoff)-projected).abs().max()),
        orthogonality_inner_product=float(((desired-projected)*projected).mean()),
        reference_informed=True,deployable=False)


def _state(n, regime, *, phase=0., amplitude=.08, frequency=None, orientation=0, separation=1):
    x = torch.arange(n, dtype=torch.float64)/n
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    if orientation:
        xx, yy = yy, xx
    # The smoke pair must straddle its L/H cutoff, or the three-channel
    # identity test degenerates to one nonzero origin before fitting starts.
    k = min(n//2-1, 5) if frequency is None else frequency
    k = max(2, k)
    if regime == "low":
        v = torch.cos(2*math.pi*xx)+.7*torch.sin(2*math.pi*yy+.23)
    elif regime in ("high_pair", "phase_cancellation"):
        v = torch.cos(2*math.pi*((k-separation)*xx+yy)) + torch.cos(2*math.pi*(k*xx+yy)+phase)
        if regime == "phase_cancellation":
            v = v + torch.cos(2*math.pi*((k-separation)*xx-yy)+.3) - torch.cos(2*math.pi*(k*xx-yy)+phase+.3)
    elif regime == "mixed":
        v = torch.cos(2*math.pi*((k-1)*xx+yy))+torch.cos(2*math.pi*(k*xx+yy)+phase)+.6*torch.cos(2*math.pi*xx+.7)
    elif regime == "localized":
        # A finite Fourier packet, no clipping or infinite Gaussian sampling.
        v = sum(torch.cos(2*math.pi*(j*xx+(j%2)*yy)+phase) for j in range(1, k+1))
    elif regime == "nearly_constant":
        amplitude = 1e-8
        v = torch.cos(2*math.pi*xx)
    else:
        raise ValueError("Unknown diagnostic field regime")
    v = v-v.mean()
    v = v/v.square().mean().sqrt()
    u = .43+float(amplitude)*v
    if bool(((u < 0) | (u > 1)).any()):
        raise ValueError("Diagnostic field violates its requested physical range; no clipping allowed")
    return u[None, None]


def _field_id(u, label):
    return label+"-"+hashlib.sha256(u.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def _model(family, track, n, *, modes=None):
    modes=max(1,n//4) if modes is None else modes
    with torch.random.fork_rng():
        torch.manual_seed(719031)
        return make_model(family, track, dict(modes=modes, split_modes=modes,
                                             width=6, depth=2)).double().eval()


def _role(family):
    if "oracle" in family:
        return "Reference-informed oracle, not deployable"
    if family in ("df", "etdrk4", "adaptive_dop853", "rf") or "fno" in family:
        return "Theirs"
    if family in ("quad2_fixed", "channel_fixed", "quad4_full", "quad2_full", "quad2_input", "analytic_quad_cubic", "even_odd_defect") or family.startswith("quadratic_"):
        return "Analytic control"
    return "Ours"


class _Experiment:
    def __init__(self, program, profile, output, protocol, check_budget):
        self.program, self.profile, self.path = program, profile, Path(output)
        self.path.mkdir(parents=True, exist_ok=True)
        self.protocol, self.check_budget = protocol or {}, check_budget
        self.rows, self.references, self.arrays = [], [], {}
        self.started = time.perf_counter()
        self.settings = self.protocol.get("diagnostics", {})
        self.n = int(self.settings.get("grid", 8 if profile == "smoke" else 16))
        self.repeats = int(self.protocol.get("timing_repeats",3 if profile == "smoke" else 5))
        self.reference_cap = int(self.settings.get("max_reference_substeps",256))
        self.uncertainty_floor = float(self.settings.get("uncertainty_floor",1e-12))
        if self.n < 8 or self.repeats < 1 or self.reference_cap < 16 or self.uncertainty_floor <= 0:
            raise ValueError("Diagnostic grid, timing repeats and reference bounds invalid")
        self.tracks = tuple(self.protocol.get("tracks", ("discrete", "continuum")))
        self.reference_cache = {}

    def save_arrays(self, prefix, **items):
        prefix = prefix.replace("/", "_")
        for name, value in items.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            self.arrays[prefix+"__"+name] = np.asarray(value)
        return prefix

    def add(self, **row):
        self.check_budget()
        row = {"diagnostic":self.program, "case_id":f"{self.program}-{len(self.rows):05d}",
               "evidence_category":"development_diagnostic", "device":"cpu", "dtype":"float64",
               "checkpoint_status":"not_applicable", **row}
        if "method" in row:
            row.setdefault("method_role", _role(row["method"]))
        self.rows.append(_clean(row))
        # A failure retains every completed result, not only console messages.
        _write(self.path/"diagnostic_rows.json", self.rows)
        return row

    def reference(self, u, h, eq, geo, track, *, parent_id):
        key = (hashlib.sha256(u.numpy().tobytes()).hexdigest(), float(h), eq.kappa,
               eq.reaction_rate, geo.grid, geo.lengths, track)
        if key in self.reference_cache:
            return self.reference_cache[key]
        self.check_budget(); start = time.perf_counter()
        base = min(max(16, math.ceil(h*eq.reaction_rate*8)),self.reference_cap//4)
        values = [lawson_reference(u,h,eq,geo,count,track,check_budget=self.check_budget)
                  for count in (base,2*base,4*base)]
        changes = [error_metrics(values[i+1]-values[i]) for i in range(2)]
        uncertainty = max(changes[-1]["error_max"], self.uncertainty_floor)
        converged = (changes[-1]["error_max"] <= max(changes[0]["error_max"]/4, self.uncertainty_floor))
        accepted = bool(converged and uncertainty <= 1e-8 and torch.isfinite(values[-1]).all())
        record = dict(reference_id=f"reference-{len(self.references):05d}",parent_id=parent_id,
            track=track,target="same-grid FD/nodal" if track=="discrete" else "same-grid dealiased Galerkin",
            N=u.shape[-1],h=h,kappa=eq.kappa,reaction_rate=eq.reaction_rate,
            refinement_counts=[base,2*base,4*base],refinement_changes=changes,
            reference_uncertainty=uncertainty,reference_accepted=accepted,
            uncertainty_is_certificate=False,reference_seconds=time.perf_counter()-start)
        self.references.append(record)
        self.reference_cache[key] = values[-1], record, values[-1]-values[-2]
        return self.reference_cache[key]

    def finalize(self, status="COMPLETED", error=None):
        _write(self.path/"diagnostic_rows.json", self.rows)
        _write(self.path/"reference_records.json", self.references)
        np.savez_compressed(self.path/"arrays.npz", **self.arrays)
        rollout_counts={}
        for row in self.rows:
            if row.get("category")=="rollout_summary":
                key="/".join(str(row.get(k,"unknown")) for k in ("track","regime","schedule_name"))
                group=rollout_counts.setdefault(key,dict(denominator=0,classifications={},statistical_scope="paired method/trajectory observations, not independent fields"))
                group["denominator"]+=1
                label=row.get("classification",row.get("status","unknown"))
                group["classifications"][label]=group["classifications"].get(label,0)+1
        summary = dict(diagnostic=self.program,status=status,rows=len(self.rows),
            reference_cases=len(self.references),unresolved_references=sum(not r["reference_accepted"] for r in self.references),
            elapsed_seconds=time.perf_counter()-self.started,device="cpu",dtype="float64",
            process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
            memory_scope="whole Python process high-water RSS, not per-method peak",
            evidence_category="bounded development; no independent confirmation",
            energy_joules=None,monetary_cost=None,analytical_uncertainty_is_certificate=False,
            initialized_neural_comparisons_are_not_trained_baselines=True,
            rollout_denominators=rollout_counts,
            configuration=dict(grid=self.n,timing_repeats=self.repeats,max_reference_substeps=self.reference_cap,
                               uncertainty_floor=self.uncertainty_floor,tracks=list(self.tracks)),error=error)
        _write(self.path/"diagnostic_summary.json", summary)
        return summary


def _ref_fields(record):
    return {k:record[k] for k in ("reference_id", "reference_uncertainty", "reference_accepted")}


def _prediction_row(ex, u, ref, record, prediction, *, method, parent_id, track, **extra):
    error = prediction-ref
    prefix = ex.save_arrays(f"row{len(ex.rows):05d}",initial=u,prediction=prediction,
        reference=ref,residual=error,residual_spectrum=torch.fft.fftn(error,dim=(-2,-1),norm="forward"))
    return ex.add(method=method,parent_id=parent_id,track=track,array_prefix=prefix,
                  **error_metrics(error),**_ref_fields(record),**extra)


def _d01(ex):
    n=ex.n;geo=Geometry((n,n),(1.,1.));eq=Equation(.004,3.);h=.12;k=n//4
    regimes=("low","high_pair","mixed","nearly_constant") if ex.profile=="smoke" else ("low","high_pair","mixed","localized","phase_cancellation","nearly_constant")
    prepared=[]
    for regime in regimes:
        u=_state(n,regime,phase=.9);parent=_field_id(u,regime)
        for track in ex.tracks:
            ref,record,delta=ex.reference(u,h,eq,geo,track,parent_id=parent)
            base=_model("df",track,n)(u,h,eq,geo)
            q=lowpass(quadratic_df_defect(u,h,eq,geo,nodes=2,target=track),k)
            prepared.append((regime,u,parent,track,ref,record,delta,base,q))
    # In-sample pooled fit is disclosed, not silently called a deployed fit.
    global_gains={}
    for track in ex.tracks:
        items=[r for r in prepared if r[3]==track]
        qq=np.concatenate([r[8].numpy().ravel() for r in items])
        dd=np.concatenate([(r[4]-r[7]).numpy().ravel() for r in items])
        global_gains[track]=scalar_oracle(qq,dd,gain_bounds("quad2_conditioned"),max(r[5]["reference_uncertainty"] for r in items))
    for regime,u,parent,track,ref,record,delta,base,q in prepared:
        defect=ref-base;unc=record["reference_uncertainty"]
        baseline=error_metrics(base-ref)["error_rms"]
        common=dict(regime=regime,h=h,N=n,output_modes=k,reference_informed=True)
        for family in ("df","quad2_fixed","quad2_full","quad4_full","quad2_conditioned","band_gain","analytic_quad_cubic"):
            prediction=_model(family,track,n)(u,h,eq,geo)
            actual={**common,"output_modes":None if family in ("df","quad2_full","quad4_full","analytic_quad_cubic") else k}
            row=_prediction_row(ex,u,ref,record,prediction,method=family,parent_id=parent,track=track,
                category="baseline",checkpoint_status="fresh_initialized_no_trained_checkpoint" if family in ("quad2_conditioned","band_gain") else "frozen_analytic",
                **actual)
            row["deployed_model_available"]=False if family in ("quad2_conditioned","band_gain") else None
        floor_prediction,floor_report=output_compression_oracle(base,ref,k)
        _prediction_row(ex,u,ref,record,floor_prediction,method="output_compression_l2_oracle",
            parent_id=parent,track=track,category="compression_floor",
            **{**common,**floor_report})
        global_gain=global_gains[track]
        if global_gain["resolved"]:
            _prediction_row(ex,u,ref,record,base+global_gain["coefficient"]*q,
                method="global_amplitude_fit",parent_id=parent,track=track,category="pooled_fit",
                gain=global_gain["coefficient"],fit_scope="in-sample diagnostic cohort; reference-informed, not validation selected",**common)
        channel=interaction_channels(u,h,eq,geo,track=track,split_modes=k,nodes=2,output_modes=k)
        dc=q.mean((-2,-1),keepdim=True).expand_as(q);lower=lowpass(q,max(1,k//2))
        cubic=lowpass(cubic_df_defect(u,h,eq,geo,nodes=4,target=track),k)
        bases={"scalar":q.numpy(),"output_band":torch.cat((dc,lower-dc,q-lower),dim=1).numpy(),
               "interaction_channel":channel.numpy(),"quadratic_cubic":torch.cat((q,cubic),dim=1).numpy()}
        for basis_name,raw in bases.items():
            raw=raw.reshape(-1,n,n)
            bound=gain_bounds("band_gain" if basis_name=="output_band" else "channel_global" if basis_name=="interaction_channel" else "quad2_conditioned")
            for bounded in (False,True):
                solved=basis_oracle(raw,defect.numpy(),bounds=bound if bounded else None,
                    uncertainty=unc,reference_difference=delta.numpy())
                response=np.einsum("c,cij->ij",np.asarray(solved["coefficients"]),raw)
                prediction=base+torch.from_numpy(response)[None,None]
                residual_rms=solved["error_rms"]
                improvement=baseline/residual_rms if record["reference_accepted"] and solved["resolved"] and residual_rms>5*unc else None
                _prediction_row(ex,u,ref,record,prediction,method=f"{basis_name}_{'bounded' if bounded else 'unrestricted'}_oracle",
                    parent_id=parent,track=track,category="oracle",basis=basis_name,bounded=bounded,
                    bounds_source="hypothetical multiplier range matched to channel gains; no deployed quadratic+cubic fit" if basis_name=="quadratic_cubic" else "actual effective architecture gain bounds",
                    oracle=solved,relative_backbone_improvement=improvement,
                    **common)


def _feature_pack(u,h,eq,geo,track):
    step=numerics.validate(u,h,geo,track)
    result={};costs={}
    for rich,name in ((False,"original"),(True,"rich")):
        start=time.perf_counter();value=physical_features(u,step,eq,geo,track,rich=rich)
        costs[name]=time.perf_counter()-start;result[name]=value.flatten().tolist()
    start=time.perf_counter()
    v=u-u.mean();spectrum=torch.fft.fftn(v,dim=(-2,-1),norm="forward")
    power=spectrum.abs().square();n=u.shape[-1];freq=torch.fft.fftfreq(n,d=1/n,dtype=u.dtype)
    radius=freq[:,None].square()+freq[None,:].square()
    energy=power.sum();result["spectral_moments"]=[float((power*radius**p).sum()/energy) for p in (1,2)]
    result["band_energies"]=[float(power[...,radius<=r*r].sum()) for r in (1,max(1,n//4))]
    costs["moments_and_bands"]=time.perf_counter()-start
    start=time.perf_counter()
    channels=interaction_channels(u,h,eq,geo,track=track,split_modes=max(1,n//4),nodes=2,output_modes=max(1,n//4))
    z=torch.fft.fftn(channels,dim=(-2,-1),norm="forward")
    result["signed_interaction"]=[float(a) for a in torch.stack((z[...,1,0].real,z[...,1,0].imag),dim=-1).flatten()]
    costs["signed_interaction"]=time.perf_counter()-start
    return result,costs


def _d02(ex):
    n=ex.n;geo=Geometry((n,n),(1.,1.));eq=Equation(.004,3.);h=.2;k=n//4
    phases=(0.,math.pi/2,math.pi) if ex.profile=="smoke" else (0.,math.pi/3,math.pi/2,math.pi,1.7*math.pi)
    for track in ex.tracks:
        cases=[]
        for phase in phases:
            u=_state(n,"mixed",phase=phase,amplitude=.11);parent=_field_id(u,"matched-power-phase")
            ref,record,_=ex.reference(u,h,eq,geo,track,parent_id=parent)
            base=_model("df",track,n)(u,h,eq,geo);q=lowpass(quadratic_df_defect(u,h,eq,geo,nodes=2,target=track),k)
            d=ref-base;bounds=gain_bounds("quad2_conditioned")
            oracle=scalar_oracle(q.numpy(),d.numpy(),bounds,record["reference_uncertainty"])
            features,costs=_feature_pack(u,h,eq,geo,track)
            optimum=oracle.get("error_rms")
            tolerance=None if optimum is None else 1.05*optimum+5*record["reference_uncertainty"]
            interval=None if tolerance is None else acceptable_gain_interval(q.numpy(),d.numpy(),tolerance,bounds)
            node_trials=[]
            for node in (.08,.15,(1-1/math.sqrt(3))/2,.30,.4):
                nq=lowpass(quadratic_df_defect(u,h,eq,geo,target=track,node_positions=[node,1-node],node_weights=[.5,.5]),k)
                fit=scalar_oracle(nq.numpy(),d.numpy(),bounds,record["reference_uncertainty"])
                node_trials.append(dict(left_node=node,fit=fit,basis=nq))
            cases.append(dict(u=u,parent=parent,ref=ref,record=record,base=base,q=q,d=d,oracle=oracle,
                              features=features,feature_cost_seconds=costs,interval=interval,phase=phase,node_trials=node_trials))
            _prediction_row(ex,u,ref,record,base+q,method="quad2_conditioned",parent_id=parent,track=track,
                category="feature_state",checkpoint_status="fresh_initialized_no_trained_checkpoint",phase=phase,
                features=features,feature_cost_seconds=costs,oracle=oracle,acceptable_gain_interval=interval,
                acceptable_interval_semantics="RMS <= 1.05 * individually optimal bounded RMS + 5 * reference uncertainty; not CI",
                node_trials=[{key:val for key,val in item.items() if key!="basis"} for item in node_trials])
        for j in range(1,len(cases)):
            a,b=cases[0],cases[j];bounds=gain_bounds("quad2_conditioned")
            q=np.concatenate([v["q"].numpy().ravel() for v in (a,b)]);d=np.concatenate([v["d"].numpy().ravel() for v in (a,b)])
            unc=max(v["record"]["reference_uncertainty"] for v in (a,b))
            shared=scalar_oracle(q,d,bounds,unc)
            separate=math.sqrt(np.mean([v["oracle"].get("error_rms",math.nan)**2 for v in (a,b)]))
            intervals=[a["interval"],b["interval"]]
            overlap=None if any(v is None for v in intervals) else max(v[0] for v in intervals)<=min(v[1] for v in intervals)
            difference={name:float(np.max(np.abs(np.asarray(a["features"][name])-np.asarray(b["features"][name])))) for name in a["features"]}
            joint=[]
            for index in range(len(a["node_trials"])):
                qa=np.concatenate([v["node_trials"][index]["basis"].numpy().ravel() for v in (a,b)])
                fit=scalar_oracle(qa,d,bounds,unc)
                joint.append(dict(left_node=a["node_trials"][index]["left_node"],fit=fit))
            finite_trials=[r for r in joint if r["fit"]["resolved"]]
            best_joint=min(finite_trials,key=lambda r:r["fit"]["error_rms"]) if finite_trials else None
            common_error=shared.get("error_rms")
            accepted=all(v["record"]["reference_accepted"] for v in (a,b))
            evidence=bool(accepted and difference["original"]<1e-10 and overlap is False and common_error is not None
                          and common_error-separate>5*unc and common_error>1.05*separate)
            individual_node_errors=[min(r["fit"]["error_rms"] for r in v["node_trials"] if r["fit"]["resolved"]) for v in (a,b)]
            separate_node_error=math.sqrt(np.mean(np.square(individual_node_errors)))
            common_node_error=best_joint["fit"]["error_rms"] if best_joint else None
            ex.add(category="feature_collision",parent_id=a["parent"],paired_parent_id=b["parent"],
                field_cluster="matched-power-phase-group",track=track,method="shared_vs_separate_gain",
                feature_differences=difference,features_matched=difference["original"]<1e-10,
                acceptable_gain_intervals=intervals,intervals_overlap=overlap,
                shared_gain=shared.get("coefficient"),individual_gains=[v["oracle"].get("coefficient") for v in (a,b)],
                shared_error_rms=common_error,separate_error_rms=separate,
                shared_over_separate=common_error/separate if common_error is not None and separate>5*unc else None,
                response_insufficiency_demonstrated=evidence,reference_uncertainty=unc,
                reference_accepted=accepted,
                response_insufficiency_scope="bounded scalar gain at frozen GL2 only, not arbitrary node choices",
                best_separate_node_gain_error_rms=separate_node_error,
                finite_node_grid_shared_over_separate=common_node_error/separate_node_error if common_node_error is not None and separate_node_error>5*unc else None,
                arbitrary_node_conditioner_insufficiency=None,
                best_shared_node_gain=best_joint,finite_node_search_not_global_node_optimum=True,
                future_solution_difference_alone_is_not_evidence=True)


def _dense_reference(ex,u,times,eq,geo,track,parent):
    """Adaptive dense output checked by tighter tolerance and independent Lawson."""
    times=np.asarray(times,float)
    def fun(t,y):
        ex.check_budget()
        return numerics.rhs(torch.from_numpy(y.copy()).reshape_as(u),eq,geo,track).numpy().ravel()
    start=time.perf_counter();answers=[]
    for rtol,atol in ((2e-10,2e-12),(2e-12,2e-14)):
        solved=solve_ivp(fun,(0,float(times[-1])),u.numpy().ravel(),method="DOP853",
                         rtol=rtol,atol=atol,t_eval=times,dense_output=True)
        if not solved.success:
            raise RuntimeError("Independent adaptive diagnostic teacher failed: "+solved.message)
        answers.append(solved)
    first,last=answers
    comparison=float(np.max(np.abs(first.y-last.y)))
    endpoint,check,_=ex.reference(u,float(times[-1]),eq,geo,track,parent_id=parent)
    independent=float(np.max(np.abs(last.y[:,-1]-endpoint.numpy().ravel())))
    uncertainty=max(comparison,independent,check["reference_uncertainty"],1e-12)
    record={**check,"reference_id":f"dense-{len(ex.references):05d}","dense_requested_times":times.tolist(),
            "adaptive_refinement_change":comparison,"independent_lawson_change":independent,
            "reference_uncertainty":uncertainty,"reference_accepted":bool(uncertainty<=1e-8 and check["reference_accepted"]),
            "adaptive_nfev":[a.nfev for a in answers],"reference_seconds":time.perf_counter()-start,
            "reference_method":"DOP853 tolerance refinement plus independently coded Lawson endpoint"}
    ex.references.append(record)
    tensor=torch.from_numpy(last.y.T.copy()).reshape(len(times),*u.shape)
    return tensor,record,last


def _rollout(model,u,schedule,eq,geo):
    values=[];value=u.clone()
    for dt in schedule:
        value=model(value,float(dt),eq,geo)
        values.append(value)
        if not bool(torch.isfinite(value).all()):
            break
    return torch.stack(values)


def _paired_latency(ex,functions):
    """Randomized paired whole-solve CPU timing; warm/cold scopes explicit."""
    rng=np.random.default_rng(814031);samples={name:[] for name in functions};cold={}
    for name,function in functions.items():
        ex.check_budget();start=time.perf_counter();function();cold[name]=time.perf_counter()-start
        function()
    for _ in range(ex.repeats):
        for name in rng.permutation(list(functions)):
            ex.check_budget();start=time.perf_counter();functions[name]();samples[name].append(time.perf_counter()-start)
    return {name:dict(latency_seconds=float(np.median(values)),latency_samples_seconds=values,
        first_invocation_seconds=cold[name],timing_scope="whole CPU solve; warm process first invocation and randomized paired warm repeats; no profiling") for name,values in samples.items()}


def _d03(ex):
    n=ex.n
    # Same continuous low field in grid slice; added modes explicitly change workload.
    designs=[("anchor",n,.12,"low"),("same_field_grid",2*n,.12,"low"),
             ("added_fine_content",2*n,.12,"high_pair"),("longer_duration",n,.3,"low")]
    for label,size,horizon,regime in designs:
        u=_state(size,regime,frequency=min(5,size//2-2));parent="D03-fixed-low-parent" if regime=="low" else "D03-added-content-parent"
        geo=Geometry((size,size),(1.,1.));eq=Equation(.004,3.)
        for track in ex.tracks:
            # A single dense accepted reference supports all exact scheduled endpoints.
            times=np.linspace(horizon/32,horizon,32)
            truths,record,dense=_dense_reference(ex,u,times,eq,geo,track,parent)
            endpoint=truths[-1]
            functions={};computed={};meta={}
            for family in ("df","etdrk4","quad2_fixed","analytic_quad_cubic"):
                model=_model(family,track,size,modes=max(1,n//4))
                for steps in ((1,2,4,8,16,32) if family in ("df","etdrk4") else (1,2,4,8)):
                    schedule=[horizon/steps]*steps;key=f"{family}/{steps}"
                    functions[key]=lambda m=model,s=schedule:_rollout(m,u,s,eq,geo)
                    computed[key]=functions[key]();meta[key]=(family,schedule)
            timings=_paired_latency(ex,functions)
            for key,prediction in computed.items():
                family,schedule=meta[key];t=np.cumsum(schedule)
                refs=torch.from_numpy(dense.sol(t).T.copy()).reshape(len(t),*u.shape)
                errors=(prediction-refs).flatten(1).square().mean(1).sqrt()
                accuracy={str(tol):bool(errors[-1]+record["reference_uncertainty"]<=tol) if record["reference_accepted"] else None for tol in (1e-4,1e-6,1e-8)}
                _prediction_row(ex,u,endpoint,record,prediction[-1],method=family,parent_id=parent,track=track,
                    category="grid_step_duration",slice=label,N=size,T=horizon,h_max=max(schedule),steps=len(schedule),schedule=schedule,
                    trajectory_max_rms=float(errors.max()),trajectory_times=t.tolist(),trajectory_error_rms=errors.tolist(),
                    endpoint_feasible_by_tolerance=accuracy,cutoff_policy="fixed physical mode cutoff" ,
                    output_modes=max(1,n//4),workload_changed=regime!="low",**timings[key])
            if size != n:
                fraction_model=_model("quad2_fixed",track,size)
                fraction_prediction=fraction_model(u,horizon,eq,geo)
                _prediction_row(ex,u,endpoint,record,fraction_prediction,method="quad2_fixed",parent_id=parent,track=track,
                    category="cutoff_contrast",slice=label,N=size,T=horizon,h_max=horizon,steps=1,schedule=[horizon],
                    cutoff_policy="fixed fraction of grid resolution; changed representation",output_modes=size//4,
                    paired_fixed_physical_modes=n//4,workload_changed=regime!="low")
            # No schedule cap can create a win: an adaptive classical endpoint is timed too.
            def adaptive():
                def fun(t,y):
                    return numerics.rhs(torch.from_numpy(y.copy()).reshape_as(u),eq,geo,track).numpy().ravel()
                return solve_ivp(fun,(0,horizon),u.numpy().ravel(),method="DOP853",rtol=1e-8,atol=1e-10)
            adaptive_answer=adaptive();start=time.perf_counter();adaptive();latency=time.perf_counter()-start
            _prediction_row(ex,u,endpoint,record,torch.from_numpy(adaptive_answer.y[:,-1]).reshape_as(u),
                method="adaptive_dop853",parent_id=parent,track=track,category="adaptive_classical",slice=label,
                N=size,T=horizon,adaptive_nfev=adaptive_answer.nfev,adaptive_steps=len(adaptive_answer.t)-1,
                schedule=np.diff(adaptive_answer.t).tolist(),latency_seconds=latency,
                timing_scope="single warm adaptive CPU whole-solve observation; lower precision than paired timing distribution")
            if track=="continuum":
                fine=[]
                for factor in (2,4):
                    lifted=fourier_resample(u,(size*factor,size*factor));fg=Geometry((size*factor,size*factor),(1.,1.))
                    answer,frec,_=ex.reference(lifted,horizon,eq,fg,track,parent_id=parent)
                    fine.append((fourier_resample(answer,(size,size)),frec))
                ex.add(category="spatial_contrast",parent_id=parent,track=track,method="same_grid_vs_projected_fine",slice=label,
                    N=size,T=horizon,spatial_error_rms=error_metrics(endpoint-fine[-1][0])["error_rms"],
                    spatial_refinement_change_rms=error_metrics(fine[-1][0]-fine[-2][0])["error_rms"],
                    spatial_reference_factors=[2,4],fine_temporal_uncertainties=[r[1]["reference_uncertainty"] for r in fine],
                    continuum_certificate=False,**_ref_fields(record))


def _d04(ex):
    n=max(ex.n,12);geo=Geometry((n,n),(1.,1.));k=n//4
    designs=[dict(label="phase0",phase=0.),dict(label="phase90",phase=math.pi/2),
        dict(label="phase180",phase=math.pi),dict(label="oriented",phase=.3,orientation=1),
        dict(label="weak_diffusion",phase=.3,kappa=.0004),dict(label="short_time",phase=.3,h=.02),
        dict(label="localized",phase=.3,regime="localized"),dict(label="mixed",phase=.3,regime="mixed"),
        dict(label="separation",phase=.3,separation=2),dict(label="cancellation",phase=.3,regime="phase_cancellation")]
    if ex.profile!="smoke":
        designs += [dict(label="frequency",phase=.3,frequency=6),dict(label="finite_amplitude",phase=.3,amplitude=.15),
                    dict(label="cancellation",phase=.3,regime="phase_cancellation"),dict(label="longer_time",phase=.3,h=.25)]
    for design in designs:
        h=design.get("h",.12);eq=Equation(design.get("kappa",.004),3.)
        u=_state(n,design.get("regime","high_pair"),phase=design["phase"],amplitude=design.get("amplitude",.08),
                 frequency=design.get("frequency",max(3,n//2-2)),orientation=design.get("orientation",0),separation=design.get("separation",1))
        parent=_field_id(u,design["label"])
        for track in ex.tracks:
            ref,record,_=ex.reference(u,h,eq,geo,track,parent_id=parent)
            for family in ("quad2_fixed","quad2_input","quad2_conditioned","band_gain","channel_fixed","channel_neural","fno_small"):
                model=_model(family,track,n);prediction=model(u,h,eq,geo)
                base=_model("df",track,n)(u,h,eq,geo)
                # The signed newly created correction coefficient, not mean/DC energy.
                mode=(0,design.get("separation",1)) if design.get("orientation",0) else (design.get("separation",1),0)
                zref=torch.fft.fftn(ref-base,dim=(-2,-1),norm="forward")[0,0,*mode].item()
                zpred=torch.fft.fftn(prediction-base,dim=(-2,-1),norm="forward")[0,0,*mode].item()
                floor=max(5*record["reference_uncertainty"],1e-12)
                phase_error=float(abs(np.angle(zpred*np.conj(zref)))) if abs(zref)>floor and abs(zpred)>floor else None
                _prediction_row(ex,u,ref,record,prediction,method=family,parent_id=parent,track=track,
                    category="signed_interaction",sweep=design,N=n,h=h,coefficient_mode=list(mode),
                    coefficient_reference=dict(real=zref.real,imag=zref.imag),
                    coefficient_predicted=dict(real=zpred.real,imag=zpred.imag),
                    absolute_coefficient_error=abs(zpred-zref),phase_error=phase_error,coefficient_scale_floor=floor,
                    coefficient_definition="Fourier coefficient of endpoint minus matched DF backbone",
                    checkpoint_status="fresh_initialized_no_trained_checkpoint" if family in ("quad2_conditioned","band_gain","channel_neural","fno_small") else "frozen_analytic",
                    local_fno_path_intact=family=="fno_small",architecture_inferiority_claim_permitted=False,
                    product_definition="nodal" if track=="discrete" else "dealiased Galerkin",
                    output_modes=k)


def _resolved_slope(x,y,uncertainty):
    x,y=np.asarray(x,float),np.asarray(y,float)
    mask=np.isfinite(y)&(y>5*float(uncertainty))&(x>0)
    if mask.sum()<3:
        return dict(slope=None,resolved_points=int(mask.sum()),reason="fewer than three above-floor scales")
    design=np.column_stack((np.log(x[mask]),np.ones(mask.sum())))
    coefficient=np.linalg.lstsq(design,np.log(y[mask]),rcond=None)[0]
    return dict(slope=float(coefficient[0]),resolved_points=int(mask.sum()),
                fitted_range=[float(x[mask].min()),float(x[mask].max())],
                fit_log_rms=float(np.sqrt(np.mean((design@coefficient-np.log(y[mask]))**2))),
                meaning="observed finite-range exponent; not an asymptotic order proof")


def _d05(ex):
    n=ex.n;geo=Geometry((n,n),(1.,1.));eq=Equation(.004,3.);background=.43;k=n//4
    v=(_state(n,"mixed",phase=.4,amplitude=.08)-background)/.08
    epsilon_values=[.02,.04,.08] if ex.profile=="smoke" else [.01,.02,.04,.08,.16]
    h_values=[.03,.06,.12] if ex.profile=="smoke" else [.015,.03,.06,.12,.24]
    for track in ex.tracks:
        parity_rows=[]
        for slice_name,values in (("amplitude",epsilon_values),("step",h_values)):
            for value in values:
                epsilon,h=(value,.12) if slice_name=="amplitude" else (.08,value)
                signed={};max_unc=0.;accepted=True
                constant=torch.full_like(v,background)
                df=_model("df",track,n)
                endpoint_background,jacobian_background,_=background_response(constant,h,eq.reaction_rate)
                background_defect=endpoint_background-df(constant,h,eq,geo)
                exact_linear=numerics.heat_step(v,h,eq,geo,track)*jacobian_background
                # The continuum DF reaction subflow is RK4, whereas the
                # Volterra formula uses the exact homogeneous response. Its
                # O(epsilon^0/epsilon^1) errors must not masquerade as missing
                # quadratic/cubic structure. Exact AD of the numerical map
                # isolates this contamination without a finite-difference step.
                with torch.enable_grad():
                    _,df_linear=torch.autograd.functional.jvp(lambda state:df(state,h,eq,geo),constant,v)
                linear_defect=exact_linear-df_linear
                for sign in (1,-1):
                    u=background+sign*epsilon*v;parent=_field_id(u,f"D05-{slice_name}")
                    ref,record,_=ex.reference(u,h,eq,geo,track,parent_id=parent);max_unc=max(max_unc,record["reference_uncertainty"])
                    accepted=accepted and record["reference_accepted"]
                    base=_model("df",track,n)(u,h,eq,geo);defect=ref-base
                    increments={count:lowpass(quadratic_df_defect(u,h,eq,geo,nodes=count,target=track),k) for count in (2,4,16,32)}
                    cubic=lowpass(cubic_df_defect(u,h,eq,geo,nodes=4,target=track),k)
                    # Compression matched for amplitude/order attribution; full control is separate.
                    signed[sign]=dict(defect=lowpass(defect-background_defect-sign*epsilon*linear_defect,k),
                        raw_defect=lowpass(defect,k),q=increments[32],cubic=cubic,reference=ref,base=base)
                    for nodes in (2,4,16,32):
                        _prediction_row(ex,u,ref,record,base+increments[nodes],method=f"quadratic_{nodes}_node",
                            parent_id=parent,track=track,category="quadrature_amplitude",slice=slice_name,
                            epsilon=epsilon,sign=sign,h=h,nodes=nodes,output_modes=k,
                            quadrature_error_rms=error_metrics(increments[nodes]-increments[32])["error_rms"],
                            quadrature_reference_nodes=32,
                            quadrature_reference_refinement_rms=error_metrics(increments[16]-increments[32])["error_rms"],
                            same_quadratic_formula=True)
                    _prediction_row(ex,u,ref,record,base+increments[32]+cubic,method="quadratic_cubic_matched_cutoff",
                        parent_id=parent,track=track,category="interaction_order",slice=slice_name,
                        epsilon=epsilon,sign=sign,h=h,output_modes=k)
                plus,minus=signed[1],signed[-1]
                even=(plus["defect"]+minus["defect"])/2
                odd=(plus["defect"]-minus["defect"])/2
                q=plus["q"];c=plus["cubic"]
                qq=float(q.square().sum());cc=float(c.square().sum())
                projection=(q*c).sum()/q.square().sum() if qq>0 else torch.tensor(0.)
                direction_fraction=float((c-projection*q).square().sum().sqrt()/c.square().sum().sqrt()) if cc>25*max_unc**2*n*n else None
                row=ex.add(category="amplitude_parity",track=track,method="even_odd_defect",parent_id="D05-paired-perturbations",
                    slice=slice_name,epsilon=epsilon,h=h,even_defect_rms=error_metrics(even)["error_rms"],
                    odd_defect_rms=error_metrics(odd)["error_rms"],even_after_quadratic_rms=error_metrics(even-q)["error_rms"],
                    odd_after_cubic_rms=error_metrics(odd-c)["error_rms"],
                    cubic_perpendicular_fraction=direction_fraction,reference_uncertainty=max_unc,
                    reference_accepted=accepted,
                    raw_even_defect_rms=error_metrics((plus["raw_defect"]+minus["raw_defect"])/2)["error_rms"],
                    raw_odd_defect_rms=error_metrics((plus["raw_defect"]-minus["raw_defect"])/2)["error_rms"],
                    background_integrator_error_rms=error_metrics(background_defect)["error_rms"],
                    linear_integrator_error_rms=error_metrics(epsilon*linear_defect)["error_rms"],
                    parity_scope="paired perturbations; subtract exact constant and AD-computed linear numerical-backbone defects; same output cutoff",
                    array_prefix=ex.save_arrays(f"parity_{len(ex.rows)}",even=even,odd=odd,quadratic=q,cubic=c))
                parity_rows.append(row)
        for slice_name,variable in (("amplitude","epsilon"),("step","h")):
            items=[r for r in parity_rows if r["slice"]==slice_name and r["reference_accepted"]]
            for metric in ("even_defect_rms","odd_defect_rms","even_after_quadratic_rms","odd_after_cubic_rms"):
                ex.add(category="observed_order",track=track,method="even_odd_defect",slice=slice_name,
                    variable=variable,metric=metric,fixed_h=.12 if slice_name=="amplitude" else None,
                    fixed_epsilon=.08 if slice_name=="step" else None,
                    reference_accepted=bool(items),
                    **_resolved_slope([r[variable] for r in items],[r[metric] for r in items],max((r["reference_uncertainty"] for r in items),default=1e-12)))


def _first_crossing(times,values,level):
    """Linear crossing interpolation; None if no crossing is observed."""
    for index in range(1,len(values)):
        if values[index-1]<level<=values[index]:
            fraction=(level-values[index-1])/(values[index]-values[index-1])
            return float(times[index-1]+fraction*(times[index]-times[index-1]))
    return float(times[0]) if values[0]>=level else None


def _trajectory_observables(states,times,eq,track):
    # states [time,1,1,N,N]; correct mean law is r(Eu-Eu²), not mean conservation.
    means=states.flatten(1).mean(1).numpy();nodal_variances=states.flatten(1).var(1,unbiased=False).numpy()
    mean_rhs=[];modes=[];variances=[]
    for state in states:
        mean_rhs.append(float(numerics.reaction_rhs(state,eq,track).mean()))
        # At a represented Nyquist mode the continuous trigonometric energy
        # and the nodal sum of squares differ. Use the declared product for
        # the physical moment identity, and preserve nodal variance separately.
        variances.append(float(numerics.product(state,state,track).mean()-state.mean().square()))
        z=torch.fft.fftn(state,dim=(-2,-1),norm="forward")[0,0,1,0].item()
        modes.append([z.real,z.imag])
    away=np.maximum(1e-14,np.abs(1-means));rates=-(np.diff(np.log(away))/np.diff(times))
    return dict(mean=means.tolist(),variance=variances,nodal_variance=nodal_variances.tolist(),mean_rhs=mean_rhs,
        variance_definition="mean of declared track product u*u minus mean(u)^2; nodal variance separately retained",
        selected_mode_complex=modes,threshold_crossing_time=_first_crossing(times,means,.8),
        observed_relaxation_rate=rates.tolist(),relaxation_definition="-delta log(abs(1-mean))/delta t; not a fitted PDE eigenvalue")


def _d06(ex):
    n=ex.n;geo=Geometry((n,n),(1.,1.));eq=Equation(.004,3.)
    schedules={"unequal_early_dense":[.005,.005,.01,.02,.04,.08,.14,.2,.2,.3],
               "uniform":[.1]*10}
    if ex.profile!="smoke":
        schedules["unfamiliar_long"]=[.005,.015,.03,.05,.1,.2,.3,.4,.4,.5,.5,.5]
    for regime in ("high_pair","localized"):
        initial=_state(n,regime,phase=.4,amplitude=.1)
        for perturb in ((0.,) if ex.profile=="smoke" else (0.,1e-3)):
            u=initial+perturb*(_state(n,"low")-.43);parent=_field_id(u,"D06-"+regime)
            for track in ex.tracks:
                for label,schedule in schedules.items():
                    times=np.concatenate(([0.],np.cumsum(schedule)))
                    truth,record,_=_dense_reference(ex,u,times[1:],eq,geo,track,parent)
                    truth=torch.cat((u[None],truth),0);reference_obs=_trajectory_observables(truth,times,eq,track)
                    for family in ("df","quad2_fixed","channel_fixed","analytic_quad_cubic"):
                        model=_model(family,track,n);start=time.perf_counter();values=_rollout(model,u,schedule,eq,geo);latency=time.perf_counter()-start
                        states=torch.cat((u[None],values),0);finite=bool(torch.isfinite(states).all() and len(states)==len(times))
                        if not finite:
                            ex.add(category="rollout_summary",method=family,parent_id=parent,track=track,
                                regime=regime,schedule_name=label,schedule=schedule,status="UNSTABLE_NONFINITE",
                                failure=True,completed_steps=len(values),**_ref_fields(record));continue
                        difference=states-truth;errors=difference.flatten(1).square().mean(1).sqrt().numpy()
                        maxerrors=difference.flatten(1).abs().max(1).values.numpy()
                        obs=_trajectory_observables(states,times,eq,track)
                        forced=torch.stack([model(truth[i],schedule[i],eq,geo) for i in range(len(schedule))])
                        forced_error=(forced-truth[1:]).flatten(1).square().mean(1).sqrt().numpy()
                        violations=int(((states<0)|(states>1)).flatten(1).any(1).sum())
                        stable=bool(not violations and np.max(np.abs(states.numpy()))<2)
                        classification=("reference_unresolved" if not record["reference_accepted"] else
                            "accurate_stable" if stable and max(errors)+record["reference_uncertainty"]<=1e-4 else "stable_inaccurate" if stable else "unstable_or_physical_range_violation")
                        prefix=ex.save_arrays(f"trajectory_{len(ex.rows)}",times=times,states=states,reference=truth,errors=difference)
                        ex.add(category="trajectory",method=family,parent_id=parent,track=track,regime=regime,
                            schedule_name=label,schedule=schedule,times=times.tolist(),T=float(times[-1]),initial_perturbation=perturb,
                            trajectory_error_rms=errors.tolist(),trajectory_error_max=maxerrors.tolist(),
                            teacher_forced_one_step_rms=forced_error.tolist(),autonomous_observables=obs,
                            reference_observables=reference_obs,array_prefix=prefix,**_ref_fields(record))
                        ex.add(category="rollout_summary",method=family,parent_id=parent,track=track,regime=regime,
                            schedule_name=label,schedule=schedule,T=float(times[-1]),steps=len(schedule),
                            error_rms=float(errors[-1]),error_max=float(maxerrors[-1]),worst_rms=float(max(errors)),
                            integrated_error=float(np.trapezoid(errors,times)),
                            relative_equilibrium_error=None if abs(1-reference_obs["mean"][-1])<1e-8 else float(errors[-1]/abs(1-reference_obs["mean"][-1])),
                            threshold_crossing_time=obs["threshold_crossing_time"],
                            reference_threshold_crossing_time=reference_obs["threshold_crossing_time"],
                            physical_range_violation_times=violations,classification=classification,failure=not stable,
                            latency_seconds=latency,criterion="accurate_stable requires worst absolute trajectory RMS <= 1e-4 and no [0,1] violation",
                            mean_conservation_not_assumed=True,**_ref_fields(record))


def _d08(ex):
    n=ex.n;geo=Geometry((n,n),(1.,1.))
    cases=[("smooth","low",.03,.004,1.),("weak_nonlinearity","high_pair",.06,.004,.05),
           ("nearly_constant","nearly_constant",.12,.004,3.),("tiny_step","mixed",.002,.004,3.),
           ("finite_amplitude","mixed",.2,.004,3.),("localized","localized",.12,.004,3.),
           ("phase_cancellation","phase_cancellation",.12,.004,3.)]
    state_cases=[]
    for label,regime,h,kappa,rate in cases:
        state_cases.append((label,_state(n,regime,phase=math.pi,amplitude=.15 if label=="finite_amplitude" else .06),h,Equation(kappa,rate),None))
    # Predeclared early roughness representatives, all ten levels live in fields stage.
    for alpha in ex.settings.get("alpha_levels",(.2,.6,1.) if ex.profile=="smoke" else tuple(i/10 for i in range(1,11))):
        parent=make_parent(910300,alpha,generator="multiscale_2d",levels=2,mean=.43,rms=.04)
        state_cases.append((f"rough_alpha_{alpha}",sample_field(parent,n),.12,Equation(.004,3.),parent.to_dict()))
    families=("df","etdrk4","quad2_fixed","quad2_conditioned","band_gain","channel_fixed","channel_neural","analytic_quad_cubic","fno_small")
    for regime,u,h,eq,parent_record in state_cases:
        parent=parent_record["parent_id"] if parent_record else _field_id(u,regime)
        for track in ex.tracks:
            ref,record,_=ex.reference(u,h,eq,geo,track,parent_id=parent)
            models={family:_model(family,track,n) for family in families}
            functions={family:lambda model=model:model(u,h,eq,geo) for family,model in models.items()}
            timing=_paired_latency(ex,functions)
            for family,model in models.items():
                pred=model(u,h,eq,geo);trained=family in ("quad2_conditioned","band_gain","channel_neural","fno_small")
                _prediction_row(ex,u,ref,record,pred,method=family,parent_id=parent,track=track,
                    category="favorable_and_stress",regime=regime,N=n,h=h,kappa=eq.kappa,reaction_rate=eq.reaction_rate,
                    checkpoint_status="fresh_initialized_no_trained_checkpoint" if trained else "frozen_analytic",
                    credible_neural_advantage_comparison=False,roughness=roughness_metrics(u),
                    continuous_parent=parent_record,reference_endpoint_roughness=roughness_metrics(ref),
                    predicted_endpoint_roughness=roughness_metrics(pred),**timing[family])
            # Same-state batching is explicitly not independent-field throughput.
            for batch in (1,4):
                batched=u.expand(batch,-1,-1,-1).contiguous()
                for family in ("quad2_fixed","channel_fixed","fno_small"):
                    model=models[family];model(batched,h,eq,geo);start=time.perf_counter();pred=model(batched,h,eq,geo);elapsed=time.perf_counter()-start
                    ex.add(category="batch_cost",method=family,parent_id=parent,track=track,regime=regime,batch=batch,
                        latency_seconds=elapsed,seconds_per_field=elapsed/batch,
                        independent_fields=1,batch_workload="same-state replicated batch; not independent-field generalization",
                        memory_bytes_input=batched.numel()*batched.element_size(),
                        batch_numerical_parity_max=float((pred[0:1]-model(u,h,eq,geo)).abs().max()),
                        credible_neural_advantage_comparison=False)
    _write(ex.path/"competitor_audit.json",dict(
        status="initialized architecture and numerical diagnostic only",
        fno_local_paths_preserved=True,
        missing_before_neural_claim=["credible training and validation selected checkpoints", "equal-data comparison",
            "equal-training-compute comparison", "accuracy-qualified equal-inference-cost frontier",
            "independent native GPU throughput and peak memory"],
        multiscale_comparator="ordinary feature-channel and multiscale competitors belong to separate bounded fitting stage",
        short_training_failure_is_not_architectural_inferiority=True))


@torch.no_grad()
def run_diagnostic(program,profile,output_dir,*,protocol=None,check_budget=lambda:None):
    """Execute a finite CPU development program and retain portable raw data.

    ``full`` is not confirmation: a launcher must freeze a separate confirmation
    protocol if development outcomes justify that expense. No existing campaign
    checkpoint is silently imported or relabeled.
    """
    if program not in PROGRAMS or profile not in ("smoke","development","full"):
        raise ValueError("Supported adjacent diagnostic ID and profile required")
    path=Path(output_dir)
    if (path/"diagnostic_summary.json").exists():
        raise FileExistsError("Preserve existing adjacent diagnostic outputs; select a fresh directory")
    ex=_Experiment(program,profile,path,protocol,check_budget)
    try:
        globals()["_"+program.lower()](ex)
    except BaseException as error:
        ex.finalize(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    return ex.finalize()
