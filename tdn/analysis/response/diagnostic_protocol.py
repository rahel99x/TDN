"""Source-bound architecture diagnostics, separate from the historical pilot gate.

Every variant has a native workload calibration. Exploratory architectural arms
receive paired fields and explicit work; they do not authorize confirmation or
silently turn the failed historical pilot into development eligibility.
"""
from __future__ import annotations

import copy
import itertools
import math

from .fields import make_parent, digest as field_digest, validate_parent_splits

PROFILES = ("diagnostic-smoke", "diagnostic")
REGIMES = ("low_mean", "high_mean", "large_amplitude", "strong_diffusion", "strong_reaction", "phase_twin")
NEW_FAMILIES = ("adaptive_temporal", "spatial_temporal")
ENDPOINTS = ("direct_fno", "residual_fno", "temporal", *NEW_FAMILIES)
TARGETS = ("fd_same_grid", "galerkin_same_grid", "galerkin_projected_fine")
DIAGNOSTIC_GPU_CASES = (
    [f"test_response_diagnostic_cuda_odd_even_parity[{n}-{f}-{t}]" for n in (6, 7) for f in NEW_FAMILIES for t in TARGETS]
    + [f"test_response_diagnostic_cuda_head_trunk_gradients[{f}-{t}]" for f in NEW_FAMILIES for t in TARGETS]
    + [f"test_response_diagnostic_cuda_physics_time_and_query_parity[{f}-{t}]" for f in NEW_FAMILIES for t in TARGETS]
    + [f"test_response_diagnostic_cuda_remainder_fp64_assembly[{f}]" for f in ENDPOINTS]
    + [f"test_response_diagnostic_cuda_frozen_gains_and_physical_nulls[{f}]" for f in ENDPOINTS]
    + [f"test_response_diagnostic_cuda_target_order_dictionary[{r}-{b}]" for r in (4, 8) for b in ("mixed", "response")]
    + [f"test_response_diagnostic_cuda_inference_cache_and_serialized_gains[{d}-{f}]" for d in ("float32", "float64") for f in NEW_FAMILIES]
    + ["test_response_diagnostic_cuda_actual_allocation"]
)


def variants(local=False):
    """One-factor controls; width reductions belong only to CPU implementation smoke."""
    width, depth = (4, 1) if local else (16, 2)
    base = dict(width=width, depth=depth, rank=4)
    full = dict(batch_mode="full_parent", objective_mode="physical",
                lr_schedule={"kind": "cosine", "minimum_ratio": .05, "warmup_updates": 1 if local else 5},
                head_warmup_updates=1 if local else 5, optimizer_eps=1e-12)
    scaled = dict(gain_normalization="frozen")
    result = []
    def add(name, family, *, model=None, training=None, primary=False, same_grid=False):
        result.append(dict(id=name, family=family, model={**base, **(model or {})},
            training=dict(training or {}), primary=primary, same_grid_only=same_grid))
    add("direct_baseline", "direct_fno", primary=True)
    add("residual_baseline", "residual_fno", primary=True)
    add("temporal_baseline", "temporal", primary=True)
    add("residual_scaled", "residual_fno", model=scaled)
    add("residual_full_batch", "residual_fno", training=full)
    add("residual_conditioned", "residual_fno", model=scaled, training=full, primary=True)
    add("adaptive_response", "adaptive_temporal", model={**scaled, "temporal_basis": "response", "time_order": "target"}, training=full)
    add("adaptive_mixed", "adaptive_temporal", model={**scaled, "temporal_basis": "mixed", "time_order": "target"}, training=full, primary=True)
    add("spatial_mixed", "spatial_temporal", model={**scaled, "temporal_basis": "mixed", "time_order": "target"}, training=full, primary=True)
    for requested in (64, 128, 210):
        add(f"adaptive_width{requested}", "adaptive_temporal",
            model={**scaled, "temporal_basis": "mixed", "time_order": "target", "width": min(requested, 8) if local else requested}, training=full)
    add("spatial_width32", "spatial_temporal", model={**scaled, "width": 8 if local else 32, "temporal_basis": "mixed", "time_order": "target"}, training=full)
    add("residual_width32", "residual_fno", model={**scaled, "width": 8 if local else 32}, training=full)
    add("adaptive_rank8", "adaptive_temporal", model={**scaled, "rank": 8, "temporal_basis": "mixed", "time_order": "target"}, training=full)
    for name, family in (("residual", "residual_fno"), ("adaptive", "adaptive_temporal")):
        extra = {**scaled, **({"temporal_basis": "mixed", "time_order": "target"} if family == "adaptive_temporal" else {})}
        add(name+"_fp64_assembly", family, model={**extra, "assembly": "fp64"}, training=full)
        add(name+"_numerical_remainder", family, model={**extra, "numerical_remainder": True}, training=full)
    add("coupled_source", "coupled_source", training=full, primary=True)
    for name, cfg in (
        ("rank2", {"rank": 2}), ("rank8", {"rank": 8}),
        ("cached", {"source_decoder": "cached_spectral"}),
        ("exponential", {"source_time_basis": "exponential"}),
        ("direct_interpolation", {"source_representation": "direct_interpolation"}),
        ("fixed_physical", {"source_representation": "fixed_physical"}),
        ("learned_rates", {"learned_time_rates": True}),
        ("numerical_remainder", {"numerical_remainder": True}),
        ("fp64_assembly", {"assembly": "fp64"}),
    ):
        add("coupled_"+name, "coupled_source", model=cfg, training=full)
    for family in ("residual_fno", "coupled_source"):
        add(family+"_rollout2", family, model=scaled if family == "residual_fno" else {},
            training={**full, "rollout_weight": .1, "rollout_steps": 2}, same_grid=True)
        add(family+"_source_supervised", family,
            model={**scaled, "auxiliary_source_head": True} if family == "residual_fno" else {},
            training={**full, "source_aux_weight": .1}, same_grid=True)
    add("coupled_consistency", "coupled_source", training={**full, "consistency_weight": .1}, same_grid=True)
    return result


def _parents(seed, split, candidate_count):
    """Prospectively declared ranges; select by reference/headroom, never learned fit."""
    parents = []
    for regime_index, regime in enumerate(REGIMES):
        for candidate in range(candidate_count):
            index = regime_index + 8*candidate
            p = make_parent(seed, split, index, max_mode=3)
            # Keep labels interpretable but provide larger physical defects than
            # the historical near-null, rapidly equilibrated tiny cohort.
            mean = {"low_mean": .12, "high_mean": .82, "large_amplitude": .5}.get(regime, .4)
            mean += ((p["seed"] % 1001)/1000-.5)*.025
            amplitude = .44 if regime == "large_amplitude" else min(mean, 1-mean)*(.65+.1*candidate)
            total = sum(abs(t["amplitude"]) for t in p["terms"])
            p["terms"] = [{**t, "amplitude": amplitude*t["amplitude"]/total} for t in p["terms"]]
            p.update(mean=mean, amplitude_bound=amplitude, kappa=.015 if regime == "strong_diffusion" else .005+.002*candidate,
                reaction_rate=25. if regime == "strong_reaction" else 8.+4*candidate,
                tiny_regime=regime, generator="tdn.response-diagnostic-fields/v1",
                parent_id=f"response-diagnostic-{split}-{regime}-{candidate}-{p['seed']:016x}",
                candidate_order=candidate, diagnostic_regime=regime,
                selection_contract="First candidate with accepted teachers and declared precision headroom; every rejection retained")
            p["translation_invariant_shape_hash"] = field_digest(dict(mean=mean,
                terms=sorted((tuple(t["mode"]), abs(t["amplitude"])) for t in p["terms"])))
            p["parent_sha256"] = field_digest({k: v for k, v in p.items() if k != "parent_sha256"})
            parents.append(p)
    return parents


def calibration_key(variant_id, target):
    return variant_id+"/"+target


def build_diagnostic_protocol(profile="diagnostic-smoke", *, targets=None, local_smoke=False,
                              calibration=None, trial_seconds=None, chunk_seconds=1500, updates=None, **unused):
    from .protocol import build_protocol, digest, GPU_TEST_CASES, budget_plan, validate_protocol
    if profile not in PROFILES or (local_smoke and profile != "diagnostic-smoke"):
        raise ValueError("Local diagnostics support only reduced implementation smoke")
    if type(chunk_seconds) is not int or not 30 <= chunk_seconds <= 1800:
        raise ValueError("Diagnostic chunks require 30..1800 science seconds")
    target_list = list(TARGETS if targets is None else targets)
    if not target_list or len(target_list) != len(set(target_list)) or not set(target_list) <= set(TARGETS):
        raise ValueError("Diagnostic targets must be unique declared equations")
    p = build_protocol("smoke", targets=target_list, local_smoke=local_smoke)
    smoke = profile == "diagnostic-smoke"
    root_seed = 116010001 if smoke else 117010001
    candidates = 1 if local_smoke else 2
    parents = [q for s in ("train", "validation") for q in _parents(root_seed, s, candidates)]
    validate_parent_splits(parents)
    tolerance = 1e-8 if local_smoke else 1e-10
    roster = variants(local_smoke)
    families = {v["family"] for v in roster}
    overrides, work = dict(trial_seconds or {}), dict(updates or {})
    if any(k not in families or type(v) is not int or not 1 <= v <= 7*24*3600 for k, v in overrides.items()):
        raise ValueError("Diagnostic allowances require known FAMILY=SECONDS, at most seven days")
    if any(k not in families or type(v) is not int or not 1 <= v <= 10_000_000 for k, v in work.items()):
        raise ValueError("Diagnostic work requires known FAMILY=COUNT")
    p.update(profile=profile, root_seed=root_seed, parents=parents,
        parent_counts={s: 6*candidates for s in ("train", "validation")},
        train_horizons=[.04, .08] if local_smoke else [.04, .06, .08, .12],
        validation_horizons=[.05] if local_smoke else [.05, .09, .14],
        development_horizons=[], seeds=[118010011] if smoke else [118010011, 118010021, 118010031],
        targets=target_list, trials={}, units={}, stages=[], unsupported_cells=[],
        diagnostic=dict(schema="tdn.response-diagnostic/v1", historical_pilot_decision="STOP_OR_UNRESOLVED",
            authorizes_development=False, authorizes_confirmation=False, roster=roster,
            primary_variants=[v["id"] for v in roster if v["primary"]],
            focus_regimes=["low_mean", "large_amplitude"], full_primary_seeds=3,
            interpretation="New architecture diagnostics, not retrospective pilot passage or fresh confirmation"),
        screening=dict(required_regimes=list(REGIMES), minimum_floor=1e-7, minimum_rms_margin=20.,
            required_mse_reduction=100., candidate_parents={t: {s: {r: [q["parent_id"] for q in parents
                if q["split"] == s and q["tiny_regime"] == r] for r in REGIMES} for s in ("train", "validation")} for t in target_list}),
        gpu_test_cases=[*GPU_TEST_CASES, *DIAGNOSTIC_GPU_CASES],
        gpu_test_files=["tests/test_response_gpu.py", "tests/test_response_diagnostic_gpu.py"])
    p["training"].update(primary_families=sorted(families), diagnostic_only=True,
        tiny_regimes=list(REGIMES), validation_every=1 if local_smoke else 40,
        record_correction_diagnostics=True, public_curve_every=1 if local_smoke else 40,
        live_every=1 if local_smoke else 40, compact_public_query_plan=True)
    p["cost"]["diagnostic_benchmark"]={"repeats":2 if local_smoke else 5,"warmup":1}
    p["reference"].update(tolerance=tolerance, max_steps=128 if local_smoke else 4096,
        max_seconds_per_query=20 if local_smoke else 180, spatial_factors=[2, 4] if local_smoke else [2, 4, 8],
        max_grid=32 if local_smoke else 256,
        projected_error_budget={"temporal_tolerances": [tolerance*.2]*(2 if local_smoke else 3), "spatial_tolerance": tolerance*.6})
    p["allowance"].update(calibrated_forecast=copy.deepcopy(calibration), trial_seconds_overrides=overrides,
        update_overrides=work, chunk_seconds=chunk_seconds,
        calibration_scope="Exact variant/target architecture, objective and batch mode; observed native 64-update joint workload")
    units = p["units"]
    def add(name, kind, deps=(), seconds=300, gpu=False, **scope):
        seconds = int(seconds)
        wall = min(2100, max(300, math.ceil((seconds+180)/60)*60))
        units[name] = dict(id=name, kind=kind, dependencies=list(dict.fromkeys(deps)), seconds=seconds,
            device="cuda" if gpu and not local_smoke else "cpu", cpus=4 if gpu else 8 if kind == "references" else 2,
            mem_gib=32, walltime=f"00:{wall//60:02d}:00", mechanism_ids=["M2" if kind in ("references", "screening") else "M3"], **scope)
        return name
    add("audit", "audit", seconds=300 if local_smoke else 900)
    add("fields", "fields", ["audit"], seconds=60)
    references = {}
    for target, split in itertools.product(target_list, ("train", "validation")):
        refs = []
        for index, parent in enumerate(q for q in parents if q["split"] == split):
            horizons = p["train_horizons"] if split == "train" else p["validation_horizons"]
            source = target != "galerkin_projected_fine"
            queries = len(horizons)*len(p["phase_variants"])
            seconds = 240 if local_smoke else 1800
            calls = 4 if source else 1
            refs.append(add(f"references-{target}-{split}-{index:03d}", "references", ["fields"], seconds=seconds,
                target_kind=target, split=split, parent_ids=[parent["parent_id"]], grid=p["train_grid"],
                horizons=horizons, source_supervision=source, requested_reference_queries=queries,
                source_teacher_calls_per_query=calls, reference_max_seconds_per_query=min(p["reference"]["max_seconds_per_query"], .8*seconds/(queries*calls))))
        references[(target, split)] = refs
    all_refs = [name for values in references.values() for name in values]
    add("screening", "screening", ["fields", *all_refs], seconds=300 if local_smoke else 900, data_stages=all_refs)
    finals = []
    for target in target_list:
        for variant in roster:
            if variant["same_grid_only"] and target == "galerkin_projected_fine":
                p["unsupported_cells"].append(dict(variant_id=variant["id"], target_kind=target,
                    status="NOT_APPLICABLE", reason="Coarse-only rollout/source identity is not justified for projected-fine hidden-mode dynamics"))
                continue
            fits = [(p["seeds"][0], "tiny_joint", "joint")]
            if not smoke:
                fits = [(seed, phase, regime) for seed in (p["seeds"] if variant["primary"] else p["seeds"][:1])
                    for phase, regime in [("tiny_joint", "joint"), *[("tiny_individual", r) for r in (REGIMES if variant["primary"] else p["diagnostic"]["focus_regimes"])]]]
            for seed, phase, regime in fits:
                requested = (2 if local_smoke else 64) if smoke else work.get(variant["family"], 2000)
                trial = dict(family=variant["family"], target_kind=target, seed=seed,
                    setting=dict(lr=.001, clip=None, loss_multiplier=1., weight_decay=0.,
                        eps=variant["training"].get("optimizer_eps",1e-8), betas=[.9, .999]),
                    phase=phase, data_count=6 if phase == "tiny_joint" else 1, updates=requested,
                    regime=regime, variant_id=variant["id"], primary=variant["primary"],
                    model=copy.deepcopy(variant["model"]), inference_benchmark=copy.deepcopy(p["cost"]["diagnostic_benchmark"]),
                    **copy.deepcopy(variant["training"]))
                if not local_smoke: trial["screen_stage"] = "screening"
                key = calibration_key(variant["id"], target)
                record = (calibration or {}).get("variant_measurements", {}).get(key)
                if smoke:
                    allowance = 240 if local_smoke else 1800
                else:
                    rate = record.get("seconds_per_update") if record else None
                    if rate is None:
                        # Planning only; binding validator refuses submission.
                        allowance = max(1500, requested*10)
                    else:
                        factor = 1. if phase == "tiny_joint" else 1/6 if trial.get("batch_mode") == "full_parent" else 1.
                        allowance = max(180, math.ceil(2*requested*rate*factor+120))
                    allowance = max(allowance, overrides.get(variant["family"], 0))
                trial.update(total_seconds=allowance, calibration_key=key,
                    calibration_workload_sha256=workload_hash(variant, target, p),
                    requested_work_scope="Logical updates; examples/forward/backward and full-batch work are separately recorded",
                    evidence_scope="primary paired seeds" if variant["primary"] else "exploratory paired architecture diagnostic")
                trial_id = "trial-"+digest(trial)[:16]
                trial["id"] = trial_id; p["trials"][trial_id] = trial
                data = [*references[(target, "train")], *references[(target, "validation")]]
                deps = [*data, "screening"]
                previous = None
                chunks = math.ceil(allowance/chunk_seconds)
                for chunk in range(chunks):
                    seconds = min(chunk_seconds, allowance-chunk*chunk_seconds)
                    previous = add(f"{trial_id}-{chunk:03d}", "train", [*deps, *([previous] if previous else [])], seconds=seconds, gpu=True,
                        trial_id=trial_id, chunk_index=chunk, chunk_count=chunks, previous_chunk=previous,
                        target_kind=target, phase=phase, data_stages=data,
                        train_stage_ids=references[(target, "train")], validation_stage_ids=references[(target, "validation")])
                finals.append(previous)
    add("diagnostic-metrics", "diagnostic_metrics", ["screening", *finals], seconds=300 if local_smoke else 900,
        registered_trial_ids=list(p["trials"]), training_stages=finals)
    add("report", "report", list(units), seconds=300 if local_smoke else 900)
    p["stages"] = list(units)
    p["gpu_stages"] = [name for name, unit in units.items() if unit["device"] == "cuda"]
    p["budget_plan"] = budget_plan(p)
    p["diagnostic"]["requested_updates"] = sum(t["updates"] for t in p["trials"].values())
    return validate_protocol(p, require_binding=False)


def workload_hash(variant, target, protocol):
    from .protocol import digest
    return digest(dict(variant=variant, target_kind=target, train_grid=protocol["train_grid"],
        horizons=protocol["train_horizons"], validation_horizons=protocol["validation_horizons"],
        phases=protocol["phase_variants"], precision=protocol["precision"],
        inference_benchmark=protocol["cost"].get("diagnostic_benchmark"),
        validation_every=protocol["training"]["validation_every"],
        diagnostic_measurements={key:protocol["training"].get(key) for key in
            ("record_correction_diagnostics","public_curve_every","live_every","compact_public_query_plan")},
        calibration_fit={"phase":"tiny_joint", "train_parents":6, "validation_parents":6,
            "selection":"declared unscreened reduced CPU bank" if protocol.get("execution_scope") == "CPU_IMPLEMENTATION_SMOKE" else "first eligible candidate per regime in sealed screening",
            "individual_forecast":"joint rate with explicit full-parent work conversion and fixed 120-second reserve; allowance is a ceiling"}))


def validate_diagnostic(p, require_binding):
    from .protocol import GPU_TEST_CASES
    from .training_control import control_config
    if p.get("gpu_test_cases") != [*GPU_TEST_CASES, *DIAGNOSTIC_GPU_CASES]:
        raise ValueError("Diagnostic native readiness requires the complete 97-case declared suite")
    if p.get("diagnostic", {}).get("authorizes_development") is not False or p["diagnostic"].get("authorizes_confirmation") is not False:
        raise ValueError("Diagnostic results cannot authorize historical development or confirmation")
    roster = {v["id"]: v for v in p["diagnostic"]["roster"]}
    if len(roster) != len(p["diagnostic"]["roster"]):
        raise ValueError("Duplicate diagnostic variant")
    for trial in p["trials"].values():
        variant = roster.get(trial["variant_id"])
        if variant is None or trial["family"] != variant["family"] or trial["model"] != variant["model"]:
            raise ValueError("Diagnostic trial differs from its declared architecture")
        control_config(trial)
        if any(trial.get(key) != value for key,value in variant["training"].items()):
            raise ValueError("Diagnostic training differs from its calibrated variant")
        if trial["setting"].get("eps") != variant["training"].get("optimizer_eps",1e-8):
            raise ValueError("Diagnostic optimizer differs from its calibrated variant")
        if p.get("execution_scope") != "CPU_IMPLEMENTATION_SMOKE" and trial.get("screen_stage") != "screening":
            raise ValueError("Native calibration and diagnostics require the same screened parent layout")
        if trial["calibration_workload_sha256"] != workload_hash(variant, trial["target_kind"], p):
            raise ValueError("Diagnostic workload calibration binding changed")
    if p["profile"] == "diagnostic" and require_binding:
        calibration = p["allowance"].get("calibrated_forecast")
        if not calibration or calibration.get("execution_mode") != "desktop-slurm" or calibration.get("verified_native_gpu") is not True:
            raise ValueError("Diagnostic submission requires a fresh native diagnostic smoke calibration")
        for trial in p["trials"].values():
            measurement = calibration.get("variant_measurements", {}).get(trial["calibration_key"])
            if (not measurement or measurement.get("workload_sha256") != trial["calibration_workload_sha256"]
                or type(measurement.get("updates")) is not int or measurement["updates"] < 64
                or not isinstance(measurement.get("seconds_per_update"), (float, int))
                or not math.isfinite(measurement["seconds_per_update"]) or measurement["seconds_per_update"] <= 0):
                raise ValueError("Every exact diagnostic variant/target needs measured matching native calibration")
