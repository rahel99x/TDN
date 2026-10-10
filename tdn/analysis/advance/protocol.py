"""Frozen fresh-cohort experiments: attribution, cheap cubic structure and exploration.

All thresholds are operational decision criteria, never theorem certificates.
Previously inspected portfolio fields are deliberately not reused as confirmation.
"""
from __future__ import annotations

import copy
import math
import random

from tdn.analysis.frontier.protocol import build_protocol as base_protocol
from tdn.research.protocol import digest

SCHEMA = "tdn.advance/v1"
PROFILES = ("smoke", "development", "full")
FAMILIES = ("df", "etdrk4", "quad2_full", "quad4_full", "analytic_quad_cubic",
    "quad2_conditioned", "commutator_raw", "commutator_cubic", "two_basis",
    "fno_legacy", "fno_zero", "fno_scaled", "fno_mean")
TRAINABLE = ("quad2_conditioned", "two_basis", "fno_legacy", "fno_zero", "fno_scaled", "fno_mean")
HYPOTHESES = {
    "A1": "Independent equations, references, mathematical limits and measurement validity",
    "A2": "Baseline-preserving and physically scaled FNO residual optimization",
    "A3": "Data, actual training compute and validation-only model selection",
    "B1": "Cheap cubic interaction shape beyond a scalar quadratic gain",
    "B2": "Mean, centered, spectral, perturbation and composition error attribution",
    "B3": "Deployable schedule transfer and complete accuracy-qualified work",
    "XH": "Causal history restores unresolved interaction information",
    "XR": "Rational/resolvent physical propagation versus exact cached alternatives",
    "XT": "Tangent response, composition and hidden-state information",
}


def _parents(profile, n):
    counts = (dict(train=64, validation=32, confirmation=64, scaling=8) if profile == "full" else
              dict(train=2, validation=2, confirmation=2, scaling=2) if profile == "smoke" else
              dict(train=8, validation=8, confirmation=16, scaling=4))
    offset = {"smoke": 81000000, "development": 82000000, "full": 83000000}[profile]
    styles = ("low", "mixed", "high_pair", "broadband", "near_nyquist", "boundary_mean",
              "reaction_stiff", "diffusion_stiff")
    high = max(2, n // 2 - 2)
    modes = {"low": [(1, 0), (0, 1)], "mixed": [(1, 0), (2, 1), (1, -2)],
        "high_pair": [(high - 1, 1), (high, 1)], "near_nyquist": [(high, 0), (0, high)],
        "boundary_mean": [(1, 0), (0, 2)], "reaction_stiff": [(1, 0), (2, 1)],
        "diffusion_stiff": [(1, 0), (high, 1)]}
    rows = []
    for si, (split, count) in enumerate(counts.items()):
        for index in range(count):
            seed = offset + si * 10000 + index
            rng = random.Random(seed); style = styles[index % 8]
            choices = [(a, b) for a in range(1, high + 1) for b in range(-min(high, 3), min(high, 3) + 1)]
            field_modes = rng.sample(choices, min(12, len(choices))) if style == "broadband" else modes[style]
            distribution = "declared_distribution"
            kappa, rate = (.001, .004, .012)[(index // 8) % 3], (1., 3., 6.)[(index // 8) % 3]
            if style == "diffusion_stiff": kappa = .04
            if style == "reaction_stiff": rate = 10.
            variance = .0006 if style == "boundary_mean" else (.002, .004, .006)[(index // 8) % 3]
            mean = .08 if style == "boundary_mean" else (.35, .5, .65)[index % 3]
            if split == "confirmation" and profile != "smoke":
                quarter = index // max(1, count // 4)
                if quarter == 1:
                    kappa *= 1.7; rate *= 1.7; distribution = "coefficient_shift"
                elif quarter == 2:
                    variance *= 2.; distribution = "amplitude_shift"
                elif quarter == 3:
                    field_modes = rng.sample(choices, min(16, len(choices)))
                    distribution = "spectral_shift"
            terms = [dict(mode=list(mode), phase=rng.uniform(0., 2 * math.pi),
                          weight=1. / math.sqrt(j + 1)) for j, mode in enumerate(field_modes)]
            # Analytic L1 bound preserves [0,1] without clipping or grid-dependent rescaling.
            rms_norm = math.sqrt(sum(t["weight"] ** 2 for t in terms) / 2)
            bound_per_rms = sum(abs(t["weight"]) for t in terms) / rms_norm
            variance = min(variance, (.9 * min(mean, 1 - mean) / bound_per_rms) ** 2)
            rows.append(dict(parent_id=f"advance-{split}-{seed}", seed=seed, field_seed=seed,
                split=split, field_cluster=f"advance-field-{seed}", continuous_field_id=f"advance-field-{seed}",
                field_terms=terms, grid=[n, n], mean=mean, variance=variance,
                spectrum=style, regime=style, category="favorable" if style == "low" else "typical" if style == "mixed" else "stress",
                distribution=distribution, kappa=kappa, reaction_rate=rate, lengths=[1., 1.],
                initial_interval_analytic_bound=math.sqrt(variance) * bound_per_rms))
    return rows


def _units(p):
    units = {}; full = p["profile"] == "full"; smoke = p["profile"] == "smoke"
    def add(name, kind, prior, seconds, path="shared", **extra):
        gpu = kind in ("train", "confirm", "scaling")
        minutes = max(10, math.ceil((seconds + 300) / 60))
        units[name] = dict(id=name, kind=kind, path=path, dependencies=prior,
            device="cuda" if gpu else "cpu", cpus=4 if gpu or kind in ("audit", "freeze", "report", "aggregate", "explore") else 8,
            mem_gib=48 if kind in ("prepare", "confirm_prepare", "scaling_prepare", "confirm", "scaling") else 32,
            walltime=f"00:{minutes:02}:00", seconds=seconds, **extra)
    add("audit", "audit", [], 300 if full else 120)
    add("diagnose", "diagnose", ["audit"], 900 if full else 120 if smoke else 300)
    add("prepare", "prepare", ["audit"], 1800 if full else 240 if smoke else 900,
        splits=["train", "validation"])
    for proto in ("XH", "XR", "XT"):
        add("explore-" + proto, "explore", ["audit"], 900 if full else 120 if smoke else 300,
            path="C", prototype_id=proto, prototype_ids=[proto])
    trains = []
    for track in p["tracks"]:
        for index, seed in enumerate(p["seeds"]):
            name = f"train-{track}-{index:03d}"; trains.append(name)
            add(name, "train", ["prepare"], 1800 if full else 240 if smoke else 900, path="AB",
                track=track, seed=seed, tracks=[track], seeds=[seed], families=list(FAMILIES),
                subset_sizes=p["training"]["subset_sizes"])
    add("freeze", "freeze", trains, 300 if full else 120)
    confirmations = []
    parents = [r["parent_id"] for r in p["parents"] if r["split"] == "confirmation"]
    for index in range(0, len(parents), p["confirmation_part_size"]):
        suffix = f"{index // p['confirmation_part_size']:03d}"; prep = "confirm-prepare-" + suffix; run = "confirm-" + suffix
        chosen = parents[index:index + p["confirmation_part_size"]]
        add(prep, "confirm_prepare", ["freeze"], 1800 if full else 240 if smoke else 900,
            parent_ids=chosen, splits=["confirmation"])
        add(run, "confirm", ["freeze", prep], 1800 if full else 300 if smoke else 900,
            parent_ids=chosen, prepared_unit=prep)
        confirmations.append(run)
    add("aggregate", "aggregate", ["freeze", *confirmations], 600 if full else 180)
    add("scaling-prepare", "scaling_prepare", ["freeze"], 1800 if full else 240 if smoke else 900,
        parent_ids=[r["parent_id"] for r in p["parents"] if r["split"] == "scaling"], splits=["scaling"])
    add("scaling", "scaling", ["freeze", "scaling-prepare"], 900 if full else 180 if smoke else 300,
        prepared_unit="scaling-prepare")
    add("report", "report", list(units), 600 if full else 240)
    return units


def build_protocol(profile="smoke"):
    if profile not in PROFILES: raise ValueError("Advance profile must be smoke, development or full")
    p = copy.deepcopy(base_protocol(profile)); full = profile == "full"; smoke = profile == "smoke"
    p.update(schema=SCHEMA, version=1, parents=_parents(profile, p["train_grid"]),
        models=list(FAMILIES), trainable_models=list(TRAINABLE), seeds=[8400011, 8400021, 8400031] if full else [8400011],
        mechanisms={k:dict(name=v,question=v,stages=[]) for k,v in HYPOTHESES.items()}, combinations={},
        primary_schedules=[[.04, .08], [.08, .16]] if not smoke else [[.02, .04], [.04, .08]],
        evaluation_horizons=[.12, .24] if not smoke else [.06, .12], endpoint_steps=[1, 2, 4, 8],
        confirmation_part_size=4 if full else 2, console_every=25,
        train_horizons=[.02,.06,.12] if not smoke else [.02,.06],
        validation_horizons=[.03,.06,.12,.24] if not smoke else [.03,.06,.12],
        timing_repeats=5 if full else 2, timing_warmup=1, bootstrap_replicates=1000 if full else 100)
    p["confirm_schedules"] = p["primary_schedules"] + [[h/k] * k for h in p["evaluation_horizons"] for k in p["endpoint_steps"]]
    p["model_config"].update(width=16 if full else 4 if smoke else 8, depth=2,
        modes=4 if full else 1 if smoke else 2, reaction_substeps=4)
    # Transport is part of each model's identity, not a tunable override.
    p["model_configs"] = {}
    p["training"].update(updates=800 if full else 2 if smoke else 40,
        tuning_updates=40 if full else 1 if smoke else 8,
        subset_sizes=[64] if full else [2] if smoke else [8],
        subset_sizes_by_family={"two_basis":[16,64],"fno_scaled":[16,64]} if full else {},
        learning_rates=[.001, .0003] if not smoke else [.001], validation_every=50 if full else 1 if smoke else 10,
        trial_seconds=90 if full else 15 if smoke else 30, tuning_seconds=15 if full else 5 if smoke else 10,
        equal_time_seconds=10 if full else 0 if smoke else 2,
        equal_time_max_updates=4000 if full else 2 if smoke else 200,
        optimizer_controls=[dict(id="default",clip_grad_norm=1.,loss_scale=1.),dict(id="unclipped_scaled",clip_grad_norm=None,loss_scale=.01)],
        normalization_floor=1e-6, loss_floor=1e-6, optimizer="AdamW",
        stopping="finite update and walltime caps; selected initialization retained; equal-time censoring explicit",
        fairness_views=["equal_data", "measured_training_compute", "equal_time_with_censoring", "equal_inference_accuracy"])
    p["selection"] = dict(split="validation", primary_family="two_basis", frozen_before_confirmation=True,
        primary_train_count=64 if full else 2 if smoke else 8,
        keep_all_declared_attribution_controls=True, batch_size=1, test_truth_selects_deployment=False,
        selection_metric="joint RMS/max tolerance on validation fields, batch-one measured cost")
    p["confirmation"] = dict(selection="two primary schedules plus each frozen validation schedule; no full Cartesian menu timing",
        primary_family="two_basis", primary_controls=["commutator_cubic", "quad4_full", "quad2_conditioned", "fno_scaled", "etdrk4"],
        observed_fields=64 if full else 2 if smoke else 16, fresh_after_freeze=True,
        confidence_interpretation="descriptive paired independent-field intervals, unadjusted; no population certificate")
    p["hypothesis_thresholds"] = dict(learned_error_ratio=1.1,practical_speedup=1.2,confidence_level=.95,
        maximum_coverage_regression=.05,required_feasible_fraction=.95,minimum_fields=5,
        primary_interval_lower_bound=1., mathematical_proof=False)
    p["diagnostics"] = dict(grid=8 if smoke else 16, horizons=[.015,.03,.06,.12], amplitudes=[.01,.02,.04,.08],
        regimes=["low", "high_pair", "broadband", "near_nyquist"], seed=8500101,
        fixed_physical_band_edges=[2.,8.], tangent_epsilon=1e-5,
        mixed_error_vectors=True, oracle_fits_are_representation_diagnostics=True)
    p["exploration"] = dict(seed={"smoke":86000000,"development":87000000,"full":88000000}[profile],
        prototype_ids=["XH","XR","XT"],confirmation_access=False,novelty="new to TDN; literature novelty not established",
        XH=dict(train_parents=24 if full else 4, test_parents=24 if full else 4, noise_levels=[0.,1e-4,1e-3],
                history="forward-generated only; acquisition charged", rollout_steps=12 if full else 3),
        XR=dict(query_counts=[1,4,16,64],relative_tolerance=1e-3),
        XT=dict(parents=24 if full else 4, finite_difference_epsilons=[1e-3,1e-4,1e-5], parity_tolerance=1e-6))
    p["portfolio_budget"] = dict(protected_exploration=True,exploration_seconds=2700 if full else 360 if smoke else 900,
        scope="separate finite prototype units; no automatic transfer or expansion", branches=["A","B","C"])
    p["scaling"].update(grids=[64,128] if full else [8,16] if smoke else [32,64],
        batches=[1,4] if not smoke else [1,2],horizon=.12 if not smoke else .06,steps=2,
        repeats=20 if full else 2 if smoke else 5,
        families=["df","etdrk4","quad4_full","commutator_cubic","two_basis","fno_scaled"],
        maximum_cases=44 if full else 20 if not smoke else 12,
        case_order="every independent parent at batch one; first complete same-physics batch at larger batch size; insufficient groups remain explicit",
        large_batch_scope="throughput subset only; no duplicated fields to pad a batch; all fields retained at batch one",
        precomputed_cpu_references=True, reference_required_for_speed_claim=True)
    p["claims"].update(neural="well-scaled local residual FNO opportunity; not a paper reproduction",
        numerical="cheap cubic interaction correction versus strongest contained analytic rule",
        exploratory="causal memory, resolvent propagation and tangent flow; finite kill tests",
        deployment="no learned policy without an underlying cost margin; oracle schedules not deployable")
    p["units"] = _units(p); p["stages"] = list(p["units"])
    p["gpu_stages"] = [name for name,u in p["units"].items() if u["device"] == "cuda"]
    p["budgets"] = {k:{key:u[key] for key in ("seconds","cpus","mem_gib","walltime")} for k,u in p["units"].items()}
    p["gpu_test_cases"] = [f"test_advance_cuda_limits[{t}-{f}]" for t in p["tracks"] for f in FAMILIES] + [
        f"test_advance_cuda_gradients[{t}-{f}]" for t in p["tracks"] for f in TRAINABLE] + [
            "test_advance_cuda_visible_allocation", "test_advance_cuda_metrics_and_memory"]
    return p


def plan_units(protocol): return list(protocol["units"].values())
def dependencies(protocol, unit): return list(protocol["units"][unit]["dependencies"])
def resource_for(protocol, unit): return {k:protocol["units"][unit][k] for k in ("cpus","mem_gib","walltime","device","seconds")}


def validate_protocol(protocol):
    from tdn.analysis.frontier.data import validate_cohorts
    if not isinstance(protocol,dict) or protocol.get("schema") != SCHEMA or protocol.get("profile") not in PROFILES:
        raise ValueError("Invalid advance scientific protocol")
    if digest(protocol) != digest(build_protocol(protocol["profile"])):
        raise ValueError("Advance declaration changed; preserve evidence and declare a new run")
    validate_cohorts(protocol); seen=set()
    for name,u in protocol["units"].items():
        wall=sum(int(x)*s for x,s in zip(u["walltime"].split(':'),(3600,60,1)))
        if name!=u["id"] or not set(u["dependencies"]) <= seen: raise ValueError("Invalid advance dependency graph")
        if not 0<u["seconds"]<=1800 or not u["seconds"]+120<=wall<=2700: raise ValueError("Unsafe advance walltime")
        if not 1<=u["cpus"]<=8 or not 1<=u["mem_gib"]<=48 or (u["device"]=="cuda" and u["cpus"]!=4):
            raise ValueError("Advance resources exceed desktop bounds")
        seen.add(name)
    if len(set(protocol["gpu_test_cases"]))!=len(protocol["gpu_test_cases"]): raise ValueError("Repeated GPU case")
    return protocol
