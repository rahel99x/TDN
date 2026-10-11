"""Frozen development portfolio; full denotes coverage, never confirmation.

All new parents are development data. Architecture/optimizer selection informed
by these experiments requires a separately frozen future confirmation protocol.
"""
from __future__ import annotations

import math
import random

from tdn.analysis.advance.protocol import build_protocol as advance_protocol
from tdn.analysis.frontier.data import validate_cohorts
from tdn.research.protocol import digest

SCHEMA = "tdn.resolve/v1"
PROFILES = ("smoke", "development", "full")
HYPOTHESES = {
    "A1": "Checked implementation, independent equations and reference uncertainty",
    "B1": "Historical cubic identities retained as implementation controls",
    "B2": "Historical mean, energy and translation checks retained",
    "R1": "Direct bounded fitting separates optimization from basis limitations",
    "R2": "Paired finite-step sweeps separate quadrature, cubic and closure errors",
    "R3": "Every tied FNO optimizer receives a bounded long opportunity",
    "R4": "Midpoint-null GL3 and shared-node transport approximate or reproduce GL4",
    "R5": "Measured complete-call work and accuracy qualify any execution benefit",
    "RX": "Noise-aware memory must beat validation-selected deterministic damping",
}


def _parents(profile, n):
    count = {"smoke": (2, 2), "development": (8, 8), "full": (64, 32)}[profile]
    offset = {"smoke": 91000000, "development": 92000000, "full": 93000000}[profile]
    high = max(2, n // 2 - 2)
    styles = ("low", "mixed", "high_pair", "overlapping_triads", "near_nyquist",
              "boundary_mean", "reaction_stiff", "diffusion_stiff")
    modes = {"low": [(1, 0), (0, 1)], "mixed": [(1, 0), (2, 0), (1, 1)],
        "high_pair": [(high - 1, 1), (high, 1)],
        "overlapping_triads": [(1, 0), (0, 1), (1, 1), (2, 0)],
        "near_nyquist": [(high, 0), (0, high)],
        "boundary_mean": [(1, 0), (0, 2), (1, 2)],
        "reaction_stiff": [(1, 0), (2, 0), (0, 1)],
        "diffusion_stiff": [(1, 0), (high, 1), (high - 1, 1)]}
    rows = []
    for si, (split, total) in enumerate(zip(("train", "validation"), count)):
        for index in range(total):
            seed = offset + si * 10000 + index; rng = random.Random(seed)
            regime = styles[index % len(styles)]
            # Independent reaction/diffusion choices, not a common time rescale.
            kappa = (.001, .004, .012)[(index // 8) % 3]
            rate = (1., 3., 6.)[((index // 8) + index % 3) % 3]
            if regime == "reaction_stiff": rate = 10.
            if regime == "diffusion_stiff": kappa = .04
            mean = .08 if regime == "boundary_mean" else (.35, .5, .65)[index % 3]
            variance = .0006 if regime == "boundary_mean" else .004
            terms = [dict(mode=list(mode), phase=rng.uniform(0., 2*math.pi), weight=1/math.sqrt(j+1))
                     for j, mode in enumerate(modes[regime])]
            bound = sum(abs(t["weight"]) for t in terms) / math.sqrt(sum(t["weight"]**2 for t in terms)/2)
            variance = min(variance, (.9*min(mean, 1-mean)/bound)**2)
            rows.append(dict(parent_id=f"resolve-{split}-{seed}", seed=seed, field_seed=seed,
                split=split, field_cluster=f"resolve-field-{seed}", continuous_field_id=f"resolve-field-{seed}",
                field_terms=terms, grid=[n,n], mean=mean, variance=variance, spectrum=regime,
                regime=regime, category="favorable" if regime=="low" else "stress",
                distribution="development_only", kappa=kappa, reaction_rate=rate, lengths=[1.,1.],
                initial_interval_analytic_bound=math.sqrt(variance)*bound))
    return rows


def _units(p):
    full, smoke = p["profile"] == "full", p["profile"] == "smoke"
    units = {}
    def add(name, kind, deps, seconds, *, gpu=False, path="AB", **scope):
        minutes=max(10, math.ceil((seconds+300)/60))
        units[name]=dict(id=name,kind=kind,dependencies=deps,seconds=seconds,
            device="cuda" if gpu else "cpu",cpus=4 if gpu or kind in ("audit","report","memory") else 8,
            mem_gib=48 if kind in ("prepare","interactions","kernels") else 32,
            walltime=f"00:{minutes:02d}:00",path=path,**scope)
    add("audit","audit",[],300 if full else 120,mechanism_ids=["A1"])
    add("prepare","prepare",["audit"],1800 if full else 240 if smoke else 900,
        splits=["train","validation"],mechanism_ids=["A1"])
    for track in p["tracks"]:
        add(f"fit-{track}","fit",["prepare"],900 if full else 240,
            track=track,seed=p["seeds"][0],mechanism_ids=["R1"])
    for track in p["tracks"]:
        for n in p["resolve"]["interactions"]["grids"]:
            for regime in p["resolve"]["interactions"]["regimes"]:
                add(f"interactions-{track}-n{n}-{regime}","interactions",["audit"],1800 if full else 240 if smoke else 900,
                    track=track,grid=n,regime=regime,mechanism_ids=["R2"])
    # Independent protection: this stage does not wait for fitting or FNO results.
    add("memory","memory",["audit"],900 if full else 240 if smoke else 450,path="C",mechanism_ids=["RX"])
    for track in p["tracks"]:
        for si,seed in enumerate(p["seeds"]):
            add(f"fairness-{track}-{si:03d}","fairness",["prepare"],1800 if full else 300 if smoke else 900,
                gpu=True,track=track,seed=seed,mechanism_ids=["R3"])
    for track in p["tracks"]:
        for n in p["resolve"]["kernels"]["grids"]:
            add(f"kernels-{track}-n{n}","kernels",["audit"],900 if full else 240 if smoke else 600,
                gpu=True,track=track,grid=n,mechanism_ids=["R4","R5"])
    add("report","report",list(units),600 if full else 240,mechanism_ids=list(HYPOTHESES))
    return units


def build_protocol(profile="smoke"):
    if profile not in PROFILES: raise ValueError("Resolve profile must be smoke, development or full")
    p=advance_protocol(profile); full=profile=="full"; smoke=profile=="smoke"
    p.update(schema=SCHEMA,version=1,scientific_scope="DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIMS",
        parents=_parents(profile,p["train_grid"]),seeds=[9400011,9400021,9400031] if full else [9400011],
        mechanisms={k:dict(name=v,question=v,stages=[]) for k,v in HYPOTHESES.items()},
        confirmation=dict(enabled=False,reason="All results inform development; new frozen confirmation required"),
        console_every=20)
    for obsolete in ("exploration","policies","screening","confirmation_part_size"):
        p.pop(obsolete,None)
    p["selection"]=dict(coefficient_fit="training only; all validation arms reported",
        neural="validation among every long/equal-time optimizer; tiny excluded",
        memory="validation only before development-test teacher generation",
        confirmation_enabled=False)
    p["claims"]=dict(scope="Development hypothesis tests only",
        numerical="GL3 midpoint identity and GL4 execution parity are separate claims",
        neural="Local optimizer/trainability audit; not a published FNO reproduction",
        deployment="No deployment policy or complete-cost advantage presumed")
    p["reference_max_substeps"]=2048; p["max_finest_substeps"]=2048
    p["resolve"]={
        "fit":dict(lower=.25,upper=1.75,peak_weight=.1,normalization_floor=1e-6,
            direct_max_iterations=80 if full else 25,direct_atol=1e-8,direct_rtol=1e-6,
            adam_updates=800 if full else 12 if smoke else 100,adam_rate=.01,
            min_validation_error_ratio=1.1),
        "interactions":dict(grids=[32,64] if full else [8] if smoke else [16],
            horizon=.12, diffusion_numbers=[.1,1.,5.] if full else [.1,1.],
            reaction_numbers=[.1,1.,3.] if full else [.1,1.],phases=[0,1] if full else [0],
            regimes=["overlapping_triads","aligned_harmonics","grid_relative"] if full else ["overlapping_triads"],
            seed=95000000+PROFILES.index(profile)*100000,amplitude=.04,mean=.4,
            reference=dict(substeps=8,max_substeps=512 if full else 128,tolerance=2e-8 if full else 2e-6,
                roundoff_floor=1e-12,spatial_factors=[2,4])),
        "fairness":dict(families=["fno_scaled","fno_mean","two_basis"],
            optimizer_settings=[dict(id="clipped",learning_rate=.001,clip_grad_norm=1.,loss_scale=1.),
                dict(id="unclipped_scaled",learning_rate=.001,clip_grad_norm=None,loss_scale=.01),
                dict(id="lower_rate_unclipped",learning_rate=.0001,clip_grad_norm=None,loss_scale=.01)],
            updates=800 if full else 2 if smoke else 80,trial_seconds=90 if full else 15 if smoke else 40,
            equal_time_seconds=10 if full else 0 if smoke else 3,equal_time_max_updates=4000 if full else 400,
            tiny_updates=200 if full else 3 if smoke else 40,tiny_seconds=15 if full else 5 if smoke else 10,
            tiny_relative_objective_target=.01,tiny_absolute_normalized_target=.01,
            train_count=64 if full else 2 if smoke else 8),
        "kernels":dict(grids=[64,128] if full else [8] if smoke else [32],
            regimes=["low","high_pair"],horizon=.06 if smoke else .12,
            schedules=[1,2],phase_replicates=2 if full else 1,
            repeats=20 if full else 2 if smoke else 5,warmup=1,reference_steps=128 if full else 32,
            seed=95500000+PROFILES.index(profile)*10000,profile_repetitions=1,
            parity_tolerance=3e-6,rms_target=2e-5,max_target=2e-5,
            reference_tolerance=2e-8 if full else 2e-6,practical_speedup=1.2),
        "memory":dict(seed=96000000+PROFILES.index(profile)*100000,
            train_parents=48 if full else 4 if smoke else 12,
            validation_parents=16 if full else 2 if smoke else 6,
            test_parents=16 if full else 2 if smoke else 6,
            max_mode=8,cutoff=2,step=.02,rollout_steps=32 if full else 3 if smoke else 12,
            noise_levels=[0.,1e-4,1e-3],noise_replicates=2 if full else 1,
            ridges=[1e-6,.01,1.] if not smoke else [1e-6,.01],
            damp_gains=[0.,.25,.5,.75,1.] if not smoke else [0.,.5,1.],
            decay_rates=[0.,1.,4.] if not smoke else [0.,4.],rate_multipliers=[1.,3.]),
    }
    p["portfolio_budget"]=dict(protected_exploration=True,exploration_seconds=900 if full else 240 if smoke else 450,
        scope="One separately allocated RX stage; no automatic expansion or transfer to training")
    p["units"]=_units(p);p["stages"]=list(p["units"])
    p["gpu_stages"]=[name for name,u in p["units"].items() if u["device"]=="cuda"]
    p["budgets"]={k:{key:u[key] for key in ("seconds","cpus","mem_gib","walltime")} for k,u in p["units"].items()}
    # Keep exact IDs synchronized with the mandatory native suite.
    p["gpu_test_cases"]=[f"test_resolve_cuda_kernel_parity[{dtype}-{track}]" for dtype in ("float32","float64") for track in p["tracks"]] + [
        f"test_resolve_cuda_kernel_{kind}[{track}]" for kind in ("nulls","gradients") for track in p["tracks"]] + [
        f"test_resolve_cuda_learning_parity[{family}-{track}]" for family in ("two_basis","fno_scaled","fno_mean") for track in p["tracks"]] + [
        f"test_resolve_cuda_zero_head_activation[{family}-{track}]" for family in ("fno_scaled","fno_mean") for track in p["tracks"]] + [
        "test_resolve_cuda_allocation","test_resolve_cuda_timing_memory"]
    return p


def plan_units(protocol): return list(protocol["units"].values())
def dependencies(protocol,unit): return list(protocol["units"][unit]["dependencies"])
def resource_for(protocol,unit): return {k:protocol["units"][unit][k] for k in ("cpus","mem_gib","walltime","device","seconds")}


def validate_protocol(protocol):
    if not isinstance(protocol,dict) or protocol.get("schema")!=SCHEMA or protocol.get("profile") not in PROFILES:
        raise ValueError("Invalid Resolve protocol")
    if digest(protocol)!=digest(build_protocol(protocol["profile"])):
        raise ValueError("Resolve declaration changed; preserve evidence and use a new run")
    validate_cohorts(protocol); seen=set()
    if any(p["split"] not in ("train","validation") for p in protocol["parents"]):
        raise ValueError("Resolve cannot relabel development data as confirmation")
    for name,u in protocol["units"].items():
        wall=sum(int(x)*s for x,s in zip(u["walltime"].split(":"),(3600,60,1)))
        if name!=u["id"] or not set(u["dependencies"])<=seen: raise ValueError("Invalid Resolve dependency graph")
        if not 0<u["seconds"]<=1800 or not u["seconds"]+120<=wall<=2700: raise ValueError("Unsafe Resolve walltime")
        if not 1<=u["cpus"]<=8 or not 1<=u["mem_gib"]<=48 or (u["device"]=="cuda" and u["cpus"]!=4):
            raise ValueError("Resolve resources exceed desktop limits")
        seen.add(name)
    if len(set(protocol["gpu_test_cases"]))!=len(protocol["gpu_test_cases"]): raise ValueError("Duplicate CUDA case")
    return protocol
