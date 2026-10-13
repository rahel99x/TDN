"""Response declarations: work-proportional trials in bounded allocations.

Profiles are separate milestones, never aliases for a single Cartesian run.
The total trial allowance is configurable; each continuation allocation remains
bounded. Walltime forecasts are planning estimates, not performance evidence.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import re

SCHEMA = "tdn.response/v1"
PROFILES = ("smoke", "pilot", "development", "confirmation", "diagnostic-smoke", "diagnostic")
TARGETS = ("fd_same_grid", "galerkin_same_grid")
GPU_TARGETS = (*TARGETS,"galerkin_projected_fine")
GPU_FAMILIES = ("residual_fno","direct_fno","temporal","coupled_source")
GPU_TEST_CASES = ([f"test_response_cuda_real_odd_even_parity[{n}-{f}-{t}]" for n in (6,7) for f in GPU_FAMILIES for t in GPU_TARGETS]
    + [f"test_response_cuda_head_and_trunk_gradients[{f}-{t}]" for f in GPU_FAMILIES for t in GPU_TARGETS]
    + [f"test_response_cuda_tensor_physics_and_query_parity[{t}]" for t in GPU_TARGETS]
    + [f"test_response_cuda_reload_cache_and_precision[{d}]" for d in ("float32","float64")]
    + [f"test_response_cuda_direct_fno_fresh_process_reload[{t}]" for t in GPU_TARGETS]
    + ["test_response_cuda_clock_and_physical_nulls","test_response_cuda_actual_allocation"]
    + [f"test_response_cuda_cached_classical_dense_and_batch[{t}]" for t in GPU_TARGETS]
    + ["test_response_cuda_competitor_auxiliary_source"]
    + [f"test_response_cuda_cached_decoder[{t}]" for t in GPU_TARGETS]
    + ["test_response_cuda_remainder_and_assembly"])
FAMILIES = ("direct_fno", "residual_fno", "temporal")
TINY_REGIMES = ("low_mean", "high_mean", "large_amplitude", "strong_diffusion", "strong_reaction", "phase_twin")
KINDS = {"audit", "fields", "references", "train", "select", "gate", "compression", "closure", "evaluate", "cost", "aggregate", "report", "numerical_controls", "screening", "diagnostic_metrics"}
MECHANISMS = {key: {"name": name} for key, name in {
    "M0": "Frozen target and execution contracts", "M1": "Mathematical identities and independent probes",
    "M2": "Resolved references and precision labels", "M3": "Diverse trainability and credible competitors",
    "M4": "Coupled response and compression", "M5": "Equivalent optimization and complete cost",
    "M6": "Frozen development evaluation", "M7": "Fresh locked confirmation", "MX": "Protected closure and observer exploration",
}.items()}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def optimizer_settings():
    return [dict(id=f"lr{lr:g}-clip{clip}-scale{scale:g}", lr=lr, clip=clip, loss_multiplier=scale,
                 weight_decay=0., eps=1e-8, betas=[.9, .999])
            for lr, clip, scale in itertools.product((1e-3, 3e-4), (None, 1.), (1., .01))]


def _safe(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def parent_declarations(seed, counts):
    result = []
    for split, count in counts.items():
        for index in range(count):
            identity = digest([seed, split, index, "parent"])
            result.append(dict(parent_id=f"response-{split}-{index:06d}-{identity[:12]}", split=split,
                               index=index, generator="tdn.response-fields/v1", dimension=2, lengths=[1., 1.]))
    return result


def build_protocol(profile="smoke", *, trial_seconds=None, chunk_seconds=1500, selected=None, freeze=None,
                   calibration=None, counts=None, targets=None, local_smoke=False, updates=None):
    if profile in ("diagnostic-smoke", "diagnostic"):
        from .diagnostic_protocol import build_diagnostic_protocol
        return build_diagnostic_protocol(profile, trial_seconds=trial_seconds, chunk_seconds=chunk_seconds,
            calibration=calibration, targets=targets, local_smoke=local_smoke, updates=updates)
    if profile not in PROFILES:
        raise ValueError("Response profile must be smoke, pilot, development or confirmation")
    smoke = profile == "smoke"
    if type(chunk_seconds) is not int or not 30 <= chunk_seconds <= 1800:
        raise ValueError("Chunk science allowance must be 30..1800 seconds")
    overrides = {} if trial_seconds is None else dict(trial_seconds)
    if any(k not in GPU_FAMILIES or type(v) is not int or not 1<=v<=7*24*3600 for k,v in overrides.items()):
        raise ValueError("Trial overrides require FAMILY=SECONDS, bounded to seven days each")
    update_overrides={} if updates is None else dict(updates)
    if any(k not in GPU_FAMILIES or type(v) is not int or not 1<=v<=10_000_000 for k,v in update_overrides.items()):
        raise ValueError("Update overrides require a known FAMILY=COUNT with a positive bounded integer")
    if local_smoke and not smoke: raise ValueError("Only implementation smoke may use the local reduced workload")
    if profile=="confirmation" and freeze:
        if counts is not None or targets is not None:
            raise ValueError("Confirmation counts and targets come only from its immutable freeze")
        counts={"confirmation":freeze["confirmation_parent_count"]}
        targets=freeze["target_contracts"]
    counts = counts or ({"train": 4, "validation": 2, "development_test": 2} if smoke else
        {"train": 32, "validation": 16, "development_test": 16} if profile == "pilot" else
        {"train": 256, "validation": 64, "development_test": 128} if profile == "development" else {"confirmation": 128})
    if any(type(v) is not int or not 1 <= v <= 4096 for v in counts.values()):
        raise ValueError("Parent counts must be positive bounded integers")
    root_seed = {"smoke": 111010001, "pilot": 112010001, "development": 113010001, "confirmation": 114010001}[profile]
    if profile=="confirmation" and freeze: root_seed=freeze["confirmation_root_seed"]
    p = dict(schema=SCHEMA, version=1, profile=profile, root_seed=root_seed, dimension=2, lengths=[1., 1.],
        scientific_scope="FROZEN_CONFIRMATION" if profile == "confirmation" else "DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIMS",
        confirmation={"enabled": profile == "confirmation", "fresh_after_freeze": True},
        mechanisms=copy.deepcopy(MECHANISMS), combinations={}, console_every=25,
        parent_counts=dict(counts), parents=parent_declarations(root_seed, counts),
        phase_variants=["original","phase_twin"], targets=list(targets or TARGETS), train_grid=8 if local_smoke else 32,
        execution_scope="CPU_IMPLEMENTATION_SMOKE" if local_smoke else "NATIVE_RESEARCH",
        grids=[8] if local_smoke else [32, 64], scaling_grids=[8] if local_smoke else [64, 128],
        train_horizons=[.02, .06] if smoke else [.005, .02, .06, .12],
        validation_horizons=[.04] if smoke else [.012, .04, .09],
        development_horizons=[.03, .08] if smoke else [.03, .08, .18, .24],
        seeds=[115010011] if smoke else [115010011, 115010021, 115010031],
        precision=dict(model="float32", reference="float64", tf32=False, output_rounding_audit=True),
        reference=dict(initial_steps=8, max_steps=128 if smoke else 4096,
            tolerance=1e-8 if smoke else 1e-10, max_seconds_per_query=20 if smoke else 180,
            spatial_factors=[2, 4], max_grid=32 if local_smoke else 128 if smoke else 512, crosscheck=True),
        training=dict(batch_size=1, validation_every=1 if smoke else 40, weight_decay=0., eps=1e-8,
            betas=[.9, .999], normalization_floor=1e-7, target_rms=2e-5, target_max=2e-5,
            checkpoint_every_updates=20, model_width=4 if local_smoke else 16, model_depth=1 if local_smoke else 2,
            primary_families=list(FAMILIES), optimizer_settings=optimizer_settings(),
            tiny_regimes=list(TINY_REGIMES), tiny_squared_error_ratio=.01, tiny_required_regimes=5,
            selected_settings_per_family=2, small_step_test_only=smoke),
        evaluation=dict(rms_target=2e-5, max_target=2e-5,tolerances=[1e-6,2e-6,1e-5,2e-5,1e-4,1e-3], ratios=[.90], peak_noninferiority=1.05,
            coverage_noninferiority=.02, error_floor=1e-12, parent_weighting="equal", query_weighting="equal within parent",
            rollout_steps=[1, 2] if smoke else [1, 2, 4, 8, 16, 32], frozen_schedules=[1, 2, 4, 8],
            comparisons=[dict(candidate="coupled_source",comparator="residual_fno",scope="primary"),
                dict(candidate="coupled_source",comparator="analytic_shared_gl4",scope="primary"),
                dict(candidate="coupled_source",comparator="direct_fno",scope="secondary"),
                dict(candidate="temporal",comparator="residual_fno",scope="secondary")], multiplicity="Holm",
            no_automatic_confirmation=True),
        cost=dict(repeats=2 if smoke else 20, warmup=1 if smoke else 3, batch_sizes=[1] if smoke else [1, 4, 16],
            synchronize=True, fresh_process=True, transfers_separate=True, p99_minimum_samples=1000),
        response=dict(compression=dict(ranks=[2, 4, 8], training_only=True,
            parent_caps={"train":32,"validation":16,"development_test":16},max_descendants_per_parent=2,
            horizons={"train":.06,"validation":.04,"development_test":.08},
            time_nodes=12 if smoke else 16 if profile=="pilot" else 24),
            closure=dict(train_parents=4 if smoke else 32, validation_parents=2 if smoke else 16,
                test_parents=2 if smoke else 16, noise_levels=[0., 1e-4, 1e-3], protected=True)),
        allowance=dict(trial_seconds_overrides=overrides, update_overrides=update_overrides, chunk_seconds=chunk_seconds,
            old_campaign_caps_overridden=True, calibrated_forecast=copy.deepcopy(calibration),
            interpretation="Allowances are ceilings, not measured runtimes or promised optimality; all continuations/attempts remain charged"),
        selection_source=copy.deepcopy(selected), freeze=copy.deepcopy(freeze), trials={}, units={}, unsupported_cells=[])
    if profile=="confirmation" and freeze:
        p["evaluation"]=copy.deepcopy(freeze["evaluation"])
    else:
        n=p["parent_counts"].get("train",0)
        for comparison in p["evaluation"]["comparisons"]:
            for key in ("candidate","comparator"):
                if not comparison[key].startswith("analytic_"):
                    comparison[key] += f"::primary::n{n}"
            comparison["seeds"]=list(p["seeds"])
            comparison["targets"]=list(p["targets"])
            comparison["grids"]=[p["train_grid"]]
    _build_units(p)
    if profile=="development":
        full_n=p["parent_counts"]["train"];seen=set()
        for trial in p["trials"].values():
            if trial["variant_id"]=="primary":continue
            candidate=f"{trial['family']}::{trial['variant_id']}::n{trial['data_count']}"
            comparator=f"{trial['family']}::primary::n{full_n}"
            if trial["family"]=="direct_fno" and trial["variant_id"]=="common_normalization":
                comparator=f"direct_fno::shared_physical_objective::n{full_n}"
            if trial["family"]=="coupled_source" and trial["variant_id"]=="clock_mean_consistency":
                comparator=f"coupled_source::clock_mean::n{full_n}"
            key=(candidate,comparator,trial["target_kind"])
            if key in seen:continue
            seen.add(key)
            p["evaluation"]["comparisons"].append(dict(candidate=candidate,comparator=comparator,
                scope="exploratory",seeds=[trial["seed"]],targets=[trial["target_kind"]],grids=[p["train_grid"]]))
        for target in p["targets"]:
            for variant in ("cached_fixed_physical","cached_direct_interpolation","cached_learned_rates"):
                p["evaluation"]["comparisons"].append(dict(candidate=f"coupled_source::{variant}::n{full_n}",
                    comparator=f"coupled_source::cached_spectral::n{full_n}",scope="exploratory",seeds=[p["seeds"][0]],
                    targets=[target],grids=[p["train_grid"]],information_contract="matched cached decoder isolates representation or learned temporal rate"))
            for variant in ("source_supervised","shared_gl4_anchor","common_normalization"):
                if target=="galerkin_projected_fine" and variant=="source_supervised":continue
                p["evaluation"]["comparisons"].append(dict(candidate=f"coupled_source::{variant}::n{full_n}",
                    comparator=f"residual_fno::{variant}::n{full_n}",scope="exploratory",seeds=[p["seeds"][0]],
                    targets=[target],grids=[p["train_grid"]],information_contract="matched supervision/anchor/normalization within named arm"))
        for pair in list(p["evaluation"]["comparisons"][:4]):
            p["evaluation"]["comparisons"].append({**copy.deepcopy(pair),"scope":"exploratory","grids":list(p["scaling_grids"]),
                "interpretation":"Same frozen weights transferred to a predeclared independent-parent subset"})
    p["stages"] = list(p["units"])
    p["gpu_stages"] = [name for name, u in p["units"].items() if u["device"] == "cuda"]
    p["gpu_test_cases"] = list(GPU_TEST_CASES)
    p["budget_plan"] = budget_plan(p)
    return validate_protocol(p, require_binding=False)


def _trial_allowance(p,family,target,updates,operation_factor=1.):
    override=p["allowance"]["trial_seconds_overrides"].get(family)
    if override is not None: return override
    if p["profile"]=="smoke": return 120 if p["execution_scope"]=="CPU_IMPLEMENTATION_SMOKE" else 600
    calibration=p["allowance"]["calibrated_forecast"] or {}
    rate=calibration.get("seconds_per_update",{}).get(family+"/"+target)
    # Uncalibrated plans remain explicitly non-submittable; this template is not a forecast.
    if rate is None: return 1500
    if not isinstance(rate,(int,float)) or not math.isfinite(rate) or rate<=0:
        raise ValueError("Invalid measured per-update calibration")
    return max(120,int(math.ceil(updates*rate*operation_factor*2+120)))


def cost_representatives(p,target):
    """Freeze timing representatives by declared inference shape, never outcomes.

    Normalization values are data-dependent constants, not a different kernel
    architecture. Aliases remain unmeasured; their accuracy/timing is not filled
    from the representative. Optimizer alternatives still compete on validation
    inside the same selected method/seed identity.
    """
    grouped={}
    for trial in p["trials"].values():
        if trial["target_kind"]!=target or trial["phase"] not in ("main","data_efficiency"):continue
        architecture={"family":trial["family"],"model":{k:v for k,v in trial["model"].items() if k!="normalization"}}
        sha=digest(architecture)
        grouped.setdefault(sha,{"architecture":architecture,"trials":[]})["trials"].append(trial)
    groups=[]
    for sha,group in grouped.items():
        identity=lambda t:f"{t['family']}::{t['variant_id']}::n{t['data_count']}"
        candidates=sorted(group["trials"],key=lambda t:(t["variant_id"]!="primary",-t["data_count"],t["seed"],identity(t),t["id"]))
        chosen=candidates[0];chosen_identity=(identity(chosen),chosen["seed"])
        representative=dict(method_id=chosen_identity[0],seed=chosen["seed"],family=chosen["family"],
            architecture_sha256=sha,architecture_config=group["architecture"],
            selection="declared primary first, largest parent count, minimum seed, stable method identity; never measured results")
        aliases=[];seen=set()
        for candidate in candidates:
            key=(identity(candidate),candidate["seed"])
            if key in seen or key==chosen_identity:continue
            seen.add(key)
            aliases.append(dict(method_id=key[0],seed=key[1],architecture_sha256=sha,
                representative_method_id=chosen_identity[0],representative_seed=chosen_identity[1],
                timing_status="NOT_MEASURED",median_seconds=None,
                scope="Informational architecture alias only; no timing, accuracy or frontier value is imputed"))
        groups.append(dict(representative=representative,aliases=aliases))
    return sorted(groups,key=lambda g:(g["representative"]["family"],g["representative"]["architecture_sha256"]))


def dense_representatives(p,target):
    candidates=[g["representative"] for g in cost_representatives(p,target) if "::primary::" in g["representative"]["method_id"]]
    expected=set((*FAMILIES,"coupled_source"))
    if {c["family"] for c in candidates}!=expected or len(candidates)!=len(expected):
        raise ValueError("Dense timing requires exactly one declared primary per family")
    return sorted(candidates,key=lambda c:c["family"])


def _build_units(p):
    units = p["units"]; smoke = p["profile"] == "smoke"
    def add(name, kind, dependencies, seconds=300, *, gpu=False, cpus=None, mechanism="M0", **scope):
        seconds = int(seconds)
        if kind=="references":
            query_count=len(scope.get("parent_ids",[]))*len(scope.get("horizons",[]))*len(scope.get("phase_variants",p["phase_variants"]))*len(scope.get("intervention_variants",["original"]))
            # An endpoint plus three source-label solves must all fit inside
            # the unit. The teacher enforces this explicit query ceiling;
            # unresolved convergence remains visible rather than dropping rows.
            teacher_calls=query_count*(4 if scope.get("source_supervision") else 1)
            cap=min(float(p["reference"]["max_seconds_per_query"]),.8*seconds/max(1,teacher_calls))
            scope["reference_max_seconds_per_query"]=cap
            scope["requested_reference_queries"]=query_count
            scope["source_teacher_calls_per_query"]=4 if scope.get("source_supervision") else 1
        # Three minutes plus clock rounding cover readiness, journaling and finalization.
        wall = min(2100, max(300, int(math.ceil((seconds+180)/60))*60))
        units[name] = dict(id=name, kind=kind, dependencies=list(dict.fromkeys(dependencies)), seconds=seconds,
            device="cuda" if gpu else "cpu", cpus=4 if gpu else (cpus or 2), mem_gib=32,
            walltime=f"00:{wall//60:02d}:00", mechanism_ids=[mechanism], **scope)
        return name
    add("audit", "audit", [], 180 if smoke else 900, mechanism="M1")
    add("fields", "fields", ["audit"], 60 if smoke else 300, mechanism="M2")
    references = {}
    for target in p["targets"]:
        short = {"fd_same_grid":"fd","galerkin_same_grid":"gal","galerkin_projected_fine":"fine"}[target]
        for split in p["parent_counts"]:
            parents = [r["parent_id"] for r in p["parents"] if r["split"] == split]
            size = 1 if target=="galerkin_projected_fine" else 2
            for index in range(0, len(parents), size):
                name = f"references-{short}-{split}-{index//size:03d}"
                add(name, "references", ["fields"], 240 if smoke else 1800, cpus=8,
                    mechanism="M2", target_kind=target, split=split, parent_ids=parents[index:index+size],
                    grid=p["train_grid"], source_supervision=p["profile"]=="development" and split in ("train","validation") and target!="galerkin_projected_fine", horizons=p["train_horizons"] if split == "train" else
                    p["validation_horizons"] if split == "validation" else p["development_horizons"])
                references.setdefault((target, split), []).append(name)
    if p["profile"] == "confirmation":
        evaluate = []
        for (target, split), refs in references.items():
            for ref in refs:
                evaluate.append(add("evaluate-"+ref.removeprefix("references-"), "evaluate", [ref],
                    1800, gpu=True, mechanism="M7", target_kind=target, data_stages=[ref], frozen_external=True))
        aggregate = add("aggregate", "aggregate", evaluate, 900, mechanism="M7")
        add("report", "report", list(units), 900, mechanism="M7")
        return
    if p["profile"] in ("smoke","development"):
        for target in p["targets"]:
            chosen=[r["parent_id"] for r in p["parents"] if r["split"]=="validation"][:2]
            prep=add("numerical-reference-"+_safe(target),"references",["fields"],240 if smoke else 1800,cpus=8,
                mechanism="M2",target_kind=target,grid=p["train_grid"],split="validation",parent_ids=chosen,
                horizons=[p["validation_horizons"][0]],panel="fixed_numerical_controls")
            add("numerical-controls-"+_safe(target),"numerical_controls",[prep],240 if smoke else 1800,gpu=True,
                mechanism="M5",target_kind=target,grid=p["train_grid"],parent_ids=chosen,
                horizons=[p["validation_horizons"][0]],data_stages=[prep],auxiliary_panels=True,
                auxiliary_reference_cap_seconds=min(p["reference"]["max_seconds_per_query"],120),auxiliary_max_grid=128,auxiliary_phase="original")
    train_units = []
    def trial(family, target, seed, setting, phase, data_count, updates, *, regime=None, selection_stage=None,
              variant_id="primary", model_overrides=None, trial_overrides=None):
        if not smoke: updates=p["allowance"]["update_overrides"].get(family,updates)
        model_config={"width":p["training"]["model_width"],"depth":p["training"]["model_depth"],"rank":4,**(model_overrides or {})}
        extra=trial_overrides or {}
        # Explicit conservative operation estimator; measured baseline includes
        # validation and I/O. This is neither measured variant latency nor optimal.
        capacity=(model_config["width"]/p["training"]["model_width"])**2*(model_config["depth"]/p["training"]["model_depth"])
        rank_factor=max(1.,model_config["rank"]/4) if family=="coupled_source" else 1.
        objective_factor=1.+(3. if extra.get("source_aux_weight") else 0.)+(3. if extra.get("consistency_weight") else 0.)+(extra.get("rollout_steps",2) if extra.get("rollout_weight") else 0.)
        validation_factor=max(1.,p["parent_counts"].get("validation",0)*len(p["validation_horizons"])/max(1,p["training"]["validation_every"])/2.)
        diagnostic_factor=max(1.,data_count*len(p["train_horizons"])/max(1,p["training"]["validation_every"])/8.)
        anchor_factor=4. if model_config.get("anchor")=="shared_gl4" else 1.
        arithmetic_factor=(8. if model_config.get("assembly")=="fp64" else 1.)*(2. if model_config.get("numerical_remainder") else 1.)
        operation_factor=capacity*rank_factor*objective_factor*anchor_factor*arithmetic_factor*max(validation_factor,diagnostic_factor)
        spec = dict(family=family, target_kind=target, seed=seed, setting=copy.deepcopy(setting), phase=phase,
            data_count=data_count, updates=updates, total_seconds=_trial_allowance(p,family,target,updates,operation_factor),
            forecast_operation_factor=operation_factor, forecast_components=dict(capacity=capacity,rank=rank_factor,objective=objective_factor,validation=validation_factor,training_diagnostics=diagnostic_factor,anchor=anchor_factor,arithmetic=arithmetic_factor),
            model=model_config, regime=regime, setting_from=selection_stage,
            variant_id=variant_id, evidence_scope="primary paired seeds" if variant_id=="primary" else "exploratory focused ablation")
        spec.update(trial_overrides or {})
        ident = "trial-"+digest(spec)[:16]; spec["id"] = ident; p["trials"][ident] = spec
        deps = references[(target, "train")] + references[(target, "validation")]
        if selection_stage: deps += [selection_stage]
        chunks = math.ceil(spec["total_seconds"] / p["allowance"]["chunk_seconds"])
        previous = None
        for index in range(chunks):
            allowance = min(p["allowance"]["chunk_seconds"], spec["total_seconds"]-index*p["allowance"]["chunk_seconds"])
            name = f"{ident}-{index:03d}"
            add(name, "train", deps + ([previous] if previous else []), allowance, gpu=True, mechanism="M3" if phase != "main" else "M4",
                trial_id=ident, chunk_index=index, chunk_count=chunks, previous_chunk=previous,
                target_kind=target, phase=phase, data_stages=deps[:len(references[(target, "train")])+len(references[(target, "validation")])],
                train_stage_ids=references[(target, "train")], validation_stage_ids=references[(target, "validation")])
            previous = name
        train_units.append(previous)
        return previous
    if p["profile"] in ("smoke", "pilot"):
        factorial = []
        settings = optimizer_settings()[:1] if smoke else optimizer_settings()
        for target, family, setting in itertools.product(p["targets"], (*FAMILIES,"coupled_source") if smoke else FAMILIES, settings):
            factorial.append(trial(family, target, p["seeds"][0], setting, "factorial", p["parent_counts"]["train"], (2 if p["execution_scope"]=="CPU_IMPLEMENTATION_SMOKE" else 64) if smoke else 400))
        selector = add("optimizer-selection", "select", factorial, 60 if smoke else 300, mechanism="M3", selection_phase="factorial", required_settings=1 if smoke else 8)
        tiny = []
        if not smoke:
            for target, family, seed in itertools.product(p["targets"], FAMILIES, p["seeds"]):
                tiny.append(trial(family,target,seed,None,"tiny_joint",6,2000,regime="joint",selection_stage=selector))
                for regime in TINY_REGIMES:
                    tiny.append(trial(family,target,seed,None,"tiny_individual",1,2000,regime=regime,selection_stage=selector))
        gate = add("gate", "gate", [selector, *tiny], 60 if smoke else 300, mechanism="M3")
        # These are diagnostic studies, independent of a favorable fit result.
        for target in p["targets"]:
            add("compression-"+_safe(target), "compression", ["audit", *sum([v for (t,s),v in references.items() if t==target], [])],
                120 if smoke else 1800, mechanism="M4", target_kind=target, grid=p["train_grid"])
        add("closure", "closure", ["audit"], 120 if smoke else 1800, mechanism="MX")
        if smoke:
            mains=[]
            for target,family in itertools.product(p["targets"],(*FAMILIES,"coupled_source")):
                mains.append(trial(family,target,p["seeds"][0],optimizer_settings()[0],"main",p["parent_counts"]["train"],2))
            selected=add("selection","select",mains,60,mechanism="M6",selection_phase="main")
            evaluated=[]
            for target in p["targets"]:
                ref=references[(target,"development_test")][0]
                evaluated.append(add("evaluate-"+_safe(target),"evaluate",[selected,ref],300,gpu=True,
                    mechanism="M6",target_kind=target,data_stages=[ref],selection_stage=selected))
                ref=references[(target,"validation")][0]
                # Native complete-call, cold-start and dense-query panels share
                # this unit; retain their work inside a bounded 33-minute job.
                cost_seconds=240 if p["execution_scope"]=="CPU_IMPLEMENTATION_SMOKE" else 1800
                add("cost-"+_safe(target),"cost",[selected,ref],cost_seconds,gpu=True,mechanism="M5",
                    target_kind=target,grid=p["train_grid"],data_stages=[ref],selection_stage=selected,
                    cost_workload="all",cost_models=dense_representatives(p,target),cost_aliases=[],
                    phase_variants=["original"],warm_horizon=p["validation_horizons"][0])
            add("aggregate","aggregate",[selected,*evaluated],180,mechanism="M6")
        add("report", "report", list(units), 300 if smoke else 900, mechanism="M6")
        return
    selection = p.get("selection_source") or {}
    roster = selection.get("selected_models", [])
    # An unbound development plan is descriptive only and cannot submit.
    if not roster:
        roster = [dict(family=f, target_kind=t, setting=optimizer_settings()[0]) for t in p["targets"] for f in (*FAMILIES,"coupled_source")]
    for item, seed in itertools.product(roster, p["seeds"]):
        trial(item["family"], item["target_kind"], seed, item.get("setting"), "main", p["parent_counts"]["train"], 3000)
    # Focused one-factor interventions, not an exhaustive Cartesian product.
    # Every declared alternative is retained as its own model identity.
    for target in p["targets"]:
        by_family={}
        for item in roster:
            if item["target_kind"]==target: by_family.setdefault(item["family"],item)
        setting=by_family["coupled_source"]["setting"]
        seed=p["seeds"][0]
        variants=[("rank2",{"rank":2},{}),("rank8",{"rank":8},{}),
            ("power_only",{"feature_mode":"power"},{}),("frozen_encoder",{"frozen_encoder":True},{}),
            ("clock_mean",{"mean_parameterization":"clock"},{}),
            ("clock_mean_consistency",{"mean_parameterization":"clock"},{"consistency_weight":.1}),
            ("exponential_time",{"source_time_basis":"exponential"},{}),
            ("polynomial_time",{"source_time_basis":"polynomial"},{}),
            ("frozen_background",{"source_feedback":"frozen_background"},{}),
            ("shared_gl4_anchor",{"anchor":"shared_gl4"},{}),
            ("source_supervised",{}, {"source_aux_weight":1.}),
            ("mean_variance_consistency",{}, {"consistency_weight":.1}),
            ("short_rollout",{}, {"rollout_weight":.1,"rollout_steps":2}),
            ("rollout4",{}, {"rollout_weight":.1,"rollout_steps":4}),
            ("common_normalization",{}, {"normalization_mode":"common"}),
            ("cached_spectral",{"source_decoder":"cached_spectral"},{}),
            ("cached_fixed_physical",{"source_decoder":"cached_spectral","source_representation":"fixed_physical"},{}),
            ("cached_direct_interpolation",{"source_decoder":"cached_spectral","source_representation":"direct_interpolation"},{}),
            ("cached_learned_rates",{"source_decoder":"cached_spectral","learned_time_rates":True},{}),
            ("numerical_remainder",{"numerical_remainder":True},{}),
            ("fp64_assembly",{"assembly":"fp64"},{})]
        for variant,model_config,trial_config in variants:
            if (variant in ("clock_mean","clock_mean_consistency") and target!="fd_same_grid") or (target=="galerkin_projected_fine" and variant in ("source_supervised","mean_variance_consistency","short_rollout","rollout4")):
                p["unsupported_cells"].append(dict(variant_id=variant,target_kind=target,status="NOT_APPLICABLE",
                    reason="This target does not justify the closed nodal identity or prepared source/rollout supervision required by this variant"))
                continue
            trial("coupled_source",target,seed,setting,"main",p["parent_counts"]["train"],3000,
                variant_id=variant,model_overrides=model_config,trial_overrides=trial_config)
        for family in ("coupled_source","direct_fno","residual_fno"):
            config=by_family[family]
            for count in (32,64,128):
                trial(family,target,seed,config["setting"],"data_efficiency",count,3000,
                    variant_id=f"data-{count}")
        trial("direct_fno",target,seed,by_family["direct_fno"]["setting"],"main",p["parent_counts"]["train"],3000,
            variant_id="shared_physical_objective",trial_overrides={"objective_mode":"shared_physical"})
        for family in ("direct_fno","residual_fno"):
            trial(family,target,seed,by_family[family]["setting"],"main",p["parent_counts"]["train"],3000,
                variant_id="common_normalization",trial_overrides={"normalization_mode":"common",**({"objective_mode":"shared_physical"} if family=="direct_fno" else {})})
            if target!="galerkin_projected_fine":
                trial(family,target,seed,by_family[family]["setting"],"main",p["parent_counts"]["train"],3000,
                    variant_id="source_supervised",model_overrides={"auxiliary_source_head":True},trial_overrides={"source_aux_weight":1.})
            if family=="residual_fno":
                trial(family,target,seed,by_family[family]["setting"],"main",p["parent_counts"]["train"],3000,
                    variant_id="shared_gl4_anchor",model_overrides={"anchor":"shared_gl4"})
            trial(family,target,seed,by_family[family]["setting"],"main",p["parent_counts"]["train"],3000,
                variant_id="standard_capacity",model_overrides={"width":32,"depth":4})
    selector = add("selection", "select", train_units, 300, mechanism="M6", selection_phase="main")
    evaluate=[]
    for target in p["targets"]:
        for ref in references[(target, "development_test")]:
            evaluate.append(add("evaluate-"+ref.removeprefix("references-"),"evaluate",[selector,ref],1800,gpu=True,
                mechanism="M6", target_kind=target, data_stages=[ref], selection_stage=selector))
    for target in p["targets"]:
        intervention_groups={"reaction":["original","reaction_interpolation","reaction_extrapolation"],
            "diffusion":["original","diffusion_interpolation","diffusion_extrapolation"],
            "amplitude":["original","amplitude_half","amplitude_near_boundary"],
            "mean":["mean_pair_low","mean_pair_high"]}
        chosen=[r["parent_id"] for r in p["parents"] if r["split"]=="development_test"][:2]
        for group,variants in intervention_groups.items():
            for index,parent in enumerate(chosen):
                ident=f"{_safe(target)}-{group}-{index:03d}"
                prep=add("intervention-reference-"+ident,"references",["fields"],1800,cpus=8,
                    mechanism="M2",target_kind=target,grid=p["train_grid"],split="development_test",parent_ids=[parent],
                    horizons=[.04,.12],phase_variants=["original"],intervention_variants=variants,
                    panel="physical_interventions",descriptive_only=True)
                evaluate.append(add("intervention-"+ident,"evaluate",[selector,prep],1800,gpu=True,
                    mechanism="M6",target_kind=target,grid=p["train_grid"],data_stages=[prep],selection_stage=selector,
                    panel="physical_interventions",descriptive_only=True,intervention_variants=variants))
        add("compression-"+_safe(target),"compression",["audit",*sum([v for (t,s),v in references.items() if t==target],[])],
            1800,mechanism="M4",target_kind=target,grid=p["train_grid"])
    add("closure","closure",["audit"],1800,mechanism="MX")
    for target,n in itertools.product(p["targets"],p["scaling_grids"]):
        # Representative timing uses separately accepted CPU-generated labels;
        # GPU allocations never acquire their primary teachers while timed.
        short=_safe(target)
        cost_parents=[r["parent_id"] for r in p["parents"] if r["split"]=="validation"][:1]
        dense_times=[p["validation_horizons"][0]*(j/16)**1.3 for j in range(1,17)]
        cost_refs=[]
        for part in range(4):
            cost_refs.append(add(f"cost-reference-{short}-n{n}-{part:03d}","references",["fields"],1800,cpus=8,
                mechanism="M2",target_kind=target,grid=n,split="validation",parent_ids=cost_parents,
                horizons=dense_times[part*4:part*4+4],phase_variants=["original"],panel="representative_timing_dense_bank"))
        groups=cost_representatives(p,target)
        for part in range(0,len(groups),2):
            selected_groups=groups[part:part+2]
            add(f"cost-warm-{short}-n{n}-{part//2:03d}","cost",[selector,*cost_refs],1800,gpu=True,
                mechanism="M5",target_kind=target,grid=n,selection_stage=selector,data_stages=cost_refs,
                cost_workload="warm",cost_models=[g["representative"] for g in selected_groups],
                cost_aliases=[a for g in selected_groups for a in g["aliases"]],phase_variants=["original"],
                warm_horizon=p["validation_horizons"][0],dense_times=dense_times,dense_query_counts=[1,2,4,8,16])
        add(f"cost-dense-{short}-n{n}","cost",[selector,*cost_refs],1800,gpu=True,
            mechanism="M5",target_kind=target,grid=n,selection_stage=selector,data_stages=cost_refs,
            cost_workload="dense",cost_models=dense_representatives(p,target),cost_aliases=[],
            phase_variants=["original"],dense_times=dense_times,dense_query_counts=[1,2,4,8,16])
        stress_parents=[r["parent_id"] for r in p["parents"] if r["split"]=="development_test"][:2]
        for index,parent in enumerate(stress_parents):
            prep=add(f"stress-reference-{short}-n{n}-{index:03d}","references",["fields"],1800,cpus=8,
                mechanism="M2",target_kind=target,grid=n,split="development_test",parent_ids=[parent],
                horizons=[.02,.08],panel="grid_relative_stress",descriptive_only=True,
                field_contract="Grid-relative modes multiplied by floor(N/8); physical field changes with grid")
            evaluate.append(add(f"stress-{short}-n{n}-{index:03d}","evaluate",[selector,prep],1800,gpu=True,
                mechanism="M6",target_kind=target,grid=n,data_stages=[prep],selection_stage=selector,
                panel="grid_relative_stress",descriptive_only=True))
        transfer=[r["parent_id"] for r in p["parents"] if r["split"]=="development_test"][:8]
        part_size=1 if n>=128 or target=="galerkin_projected_fine" else 2
        for index in range(0,len(transfer),part_size):
            prep=add(f"transfer-reference-{short}-n{n}-{index//part_size:03d}","references",["fields"],1800,cpus=8,
                mechanism="M2",target_kind=target,grid=n,split="development_test",parent_ids=transfer[index:index+part_size],
                horizons=[.04,.12],panel="paired_grid_transfer")
            evaluate.append(add(f"transfer-{short}-n{n}-{index//part_size:03d}","evaluate",[selector,prep],1800,gpu=True,
                mechanism="M6",target_kind=target,grid=n,data_stages=[prep],selection_stage=selector,
                panel="same_weights32_to_grid; fixed continuous fields, not grid-relative stress"))
    add("aggregate","aggregate",[selector,*evaluate],900,mechanism="M6")
    add("report","report",list(units),900,mechanism="M6")


def budget_plan(p):
    wall=lambda u:sum(int(v)*s for v,s in zip(u["walltime"].split(":"),(3600,60,1)))
    reference_units=[u for u in p["units"].values() if u["kind"]=="references"]
    # Seven saved fields: two float32 and five float64. Source-supervised
    # queries additionally retain three float64 fields and their time values.
    # This is a declared uncompressed numeric payload, not a filesystem bound.
    reference_bytes=sum(u["requested_reference_queries"]*(u["grid"]**p["dimension"]*
        (48+(24 if u.get("source_supervision") else 0))+(24 if u.get("source_supervision") else 0))
        for u in reference_units)
    return dict(jobs=len(p["units"]),gpu_jobs=sum(u["device"]=="cuda" for u in p["units"].values()),
        allocation_seconds=sum(wall(u) for u in p["units"].values()),
        gpu_allocation_seconds=sum(wall(u) for u in p["units"].values() if u["device"]=="cuda"),
        numerical_seconds=sum(u["seconds"] for u in p["units"].values()),parents=p["parent_counts"],trials=len(p["trials"]),
        estimated_array_bytes=sum(p["parent_counts"].values())*len(p["targets"])*len(p["train_horizons"])*len(p["phase_variants"])*p["train_grid"]**2*8*8,
        artifact_estimate_scope="Nominal primary-array estimate only, not a worst-case storage bound; see artifact_storage for declared reference payload and unbounded components",
        artifact_storage=dict(worst_case_bytes=None,worst_case_status="UNAVAILABLE_NOT_A_FILESYSTEM_LIMIT",
            declared_reference_queries=sum(u["requested_reference_queries"] for u in reference_units),
            declared_uncompressed_reference_array_bytes=reference_bytes,
            reference_scope="All declared reference grids, phase/intervention descendants and source-label arrays; excludes NPZ/JSON/container overhead and transient finer teacher work arrays",
            checkpoint_continuation_units=sum(u["kind"]=="train" for u in p["units"].values()),
            unestimated_components=["model/optimizer/initial/selected checkpoint tensors and their container overhead",
                "cumulative histories duplicated across exact continuation journals",
                "evaluation predictions, profiling traces, diagnostics, rows, plots and logs",
                "temporary atomic writes, export archives and review parts"],
            recovery_scope="Every retained attempt and recovery copy adds storage; retries are not bounded by the original protocol",
            actual_bytes=None,interpretation="Planning payload is not measured disk use or a guaranteed upper bound; inspect free disk and actual stage inventories during execution"),
        runtime_forecast=None if p["allowance"]["calibrated_forecast"] is None else p["allowance"]["calibrated_forecast"],
        interpretation="Summed allocation ceilings; completed trials make later continuation chunks no-ops, but their allocations remain recorded")


def ancestors(p, stage):
    found=set()
    def visit(name):
        for parent in p["units"][name]["dependencies"]:
            if parent not in found: found.add(parent); visit(parent)
    visit(stage)
    return [name for name in p["stages"] if name in found]


def plan_units(p): return [p["units"][name] for name in p["stages"]]
def dependencies(p, unit): return list(p["units"][unit]["dependencies"])
def resource_for(p, unit): return {k:p["units"][unit][k] for k in ("cpus","mem_gib","walltime","device","seconds")}


def validate_protocol(p, *, require_binding=True):
    if not isinstance(p,dict) or p.get("schema")!=SCHEMA or p.get("profile") not in PROFILES:
        raise ValueError("Invalid Response declaration")
    if p["profile"] in ("diagnostic-smoke", "diagnostic"):
        from .diagnostic_protocol import validate_diagnostic
        validate_diagnostic(p, require_binding)
    if p["profile"] in ("pilot","development") and require_binding:
        calibration=p.get("allowance",{}).get("calibrated_forecast")
        if not calibration or calibration.get("execution_mode")!="desktop-slurm" or not calibration.get("verified_native_gpu"):
            raise ValueError("Pilot/development submission requires measured native calibration; CPU smoke is not calibration")
        required={(t["family"],t["target_kind"]) for t in p["trials"].values()}
        rates=calibration.get("seconds_per_update",{})
        if any(not isinstance(rates.get(f+"/"+t),(int,float)) or not math.isfinite(rates[f+"/"+t]) or rates[f+"/"+t]<=0 for f,t in required):
            raise ValueError("Every declared family/target requires measured native calibration")
    if p["profile"]=="development" and require_binding:
        selected=p.get("selection_source")
        if not selected or selected.get("gate_decision")!="CONTINUE" or not selected.get("selected_models"):
            raise ValueError("Development requires an eligible non-smoke pilot gate and explicit selected model protocol")
    if p["profile"]=="confirmation" and require_binding:
        frozen=p.get("freeze")
        if not frozen or not re.fullmatch(r"[0-9a-f]{64}", frozen.get("sha256","")):
            raise ValueError("Confirmation requires an immutable separately located freeze artifact")
    if not p.get("targets") or not set(p["targets"]) <= {"fd_same_grid","galerkin_same_grid","galerkin_projected_fine"}:
        raise ValueError("Unknown equation target")
    expected="FROZEN_CONFIRMATION" if p["profile"]=="confirmation" else "DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIMS"
    if p.get("scientific_scope")!=expected or p.get("confirmation",{}).get("enabled")!=(p["profile"]=="confirmation"):
        raise ValueError("Response scope cannot relabel development as confirmation")
    order=p.get("stages",[])
    if len(order)!=len(set(order)) or set(order)!=set(p.get("units",{})):
        raise ValueError("Response stage order must name each declared unit exactly once")
    seen=set()
    for name in order:
        u=p["units"][name]
        wall=sum(int(v)*s for v,s in zip(u["walltime"].split(":"),(3600,60,1)))
        if name!=u["id"] or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",name) or u["kind"] not in KINDS:
            raise ValueError("Invalid Response unit")
        if not set(u["dependencies"])<=seen or len(u["dependencies"])!=len(set(u["dependencies"])):
            raise ValueError("Response dependencies must form an ordered acyclic graph")
        if not 0<u["seconds"]<=1800 or not u["seconds"]+120<=wall<=2100:
            raise ValueError("Response scientific deadline must leave shutdown headroom within35minutes")
        if u["device"] not in ("cpu","cuda") or not 1<=u["cpus"]<=8 or not 1<=u["mem_gib"]<=48 or (u["device"]=="cuda" and u["cpus"]>4):
            raise ValueError("Response resources exceed desktop envelope")
        if u["kind"]=="train" and u.get("trial_id") not in p["trials"]:
            raise ValueError("Training unit lacks a declared trial")
        seen.add(name)
    if not seen or p["units"].get("report",{}).get("kind")!="report":raise ValueError("Failure-aware report required")
    if set(p["units"]["report"]["dependencies"])!=seen-{"report"}:raise ValueError("Report must observe every declared stage")
    ids=[r["parent_id"] for r in p["parents"]]
    if len(ids)!=len(set(ids)):raise ValueError("Duplicate independent parent IDs")
    digest(p)
    return p
