"""Immutable, stdlib-readable portfolio declarations and execution DAG.

The full protocol is declared before any new confirmation is generated. The
55/25/20 allocation covers discretionary A/B/C probe work; shared independent
teachers, audit and confirmation have separately enumerated finite budgets.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random

from tdn.analysis.frontier.protocol import build_protocol as frontier_protocol

SCHEMA = "tdn.portfolio/v1"
PROFILES = ("smoke", "development", "full")
A_FAMILIES = ("historical_half", "quad2_fixed", "quad2_amplitude", "quad2_nodes",
    "quad2_joint", "quad2_linear", "quad2_conditioned", "quad2_input", "quad2_full",
    "quad2_conditioned_full", "quad4_fixed", "quad4_full", "quad4_conditioned",
    "analytic_quad_cubic", "df", "rf", "etdrk4", "fno_small", "fno_standard", "direct_fno")
B_FAMILIES = ("residual_quad2", "conditioned_rich", "band_gain")
FAMILIES = A_FAMILIES + B_FAMILIES
FROZEN = ("historical_half", "quad2_fixed", "quad2_input", "quad2_full", "quad4_fixed", "quad4_full",
          "analytic_quad_cubic", "df", "rf", "etdrk4")
TRAINABLE = tuple(f for f in FAMILIES if f not in FROZEN)
HYPOTHESES = {
    "A1": "Normalization against the preserved half-normalized control",
    "A2": "Amplitude, node, linear and neural conditioning attribution",
    "A3": "Matched node counts and compression placement",
    "A4": "Equivalent endpoint menus and physical backbone controls",
    "A5": "Credible FNO optimization and distinct fairness objectives",
    "A6": "Effective learned parameters and correction identifiability",
    "B1": "Cheap enriched physical conditioning",
    "B2": "Learn only a residual of normalized analytic quadrature",
    "B3": "Band-dependent correction and compression tradeoff",
    "B4": "Measured overhead, reuse and accuracy-qualified complete cost",
    "B5": "Loss scaling, data efficiency and paired generalization",
    "X01": "Reusable state encoding and continuous-time interaction kernels",
    "X02": "Coarse-state indistinguishability and unresolved-scale memory",
    "X03": "Select interactions before products and distill selection rules",
}
GPU_KINDS = ("train", "confirm", "scaling")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def endpoint_schedules(protocol):
    """Exactly the same endpoint opportunities for every applicable method."""
    schedules = []
    for final in protocol["evaluation_horizons"]:
        for count in protocol["endpoint_steps"]:
            schedules.append([final / count] * count)
    schedules.extend(protocol["primary_schedules"])
    unique = {}
    for schedule in schedules:
        key = tuple(round(h, 12) for h in schedule)
        unique.setdefault(key, list(schedule))
    return list(unique.values())


def _unit(name, kind, path, prior, seconds, *, cpus=4, **extra):
    gpu = kind in GPU_KINDS
    minutes = min(45, max(10, math.ceil(seconds / 60) + (10 if gpu else 8)))
    return dict(id=name, kind=kind, path=path, dependencies=list(prior),
        device="cuda" if gpu else "cpu", cpus=4 if gpu else cpus,
        mem_gib=48 if kind in ("prepare", "confirm_prepare", "confirm", "scaling", "diagnose") else 32,
        walltime=f"00:{minutes:02d}:00", seconds=seconds, **extra)


def _units(p):
    profile = p["profile"]; full = profile == "full"; smoke = profile == "smoke"
    units = {}
    def add(name, kind, path, prior, seconds, **kw):
        units[name] = _unit(name, kind, path, prior, seconds, **kw)
    add("audit", "audit", "shared", [], 300 if full else 120)
    add("diagnose", "diagnose", "shared", ["audit"], 1200 if full else 300 if smoke else 600, cpus=8)
    add("prepare", "prepare", "shared", ["audit"], 1800 if full else 300 if smoke else 900, cpus=8,
        splits=["train", "validation"])
    probe = p["portfolio_budget"]["probe_seconds"]
    for index, hypothesis in enumerate(("X01", "X02", "X03")):
        add(f"explore-{hypothesis}", "explore", "C", ["audit"], probe["C"] // 3,
            prototype_ids=[hypothesis])
    trains = []
    for path, families in (("A", A_FAMILIES), ("B", B_FAMILIES)):
        for index, seed in enumerate(p["seeds"]):
            name = f"train-{path}-{index:03d}"; trains.append(name)
            add(name, "train", path, ["audit", "prepare"], probe[path] // len(p["seeds"]),
                families=list(families), seeds=[seed], tracks=p["tracks"],
                subset_sizes=p["training"]["subset_sizes"])
    add("freeze", "freeze", "shared", trains, 300 if full else 120)
    confirmations = []
    parents = [r["parent_id"] for r in p["parents"] if r["split"] == "confirmation"]
    for index in range(0, len(parents), p["confirmation_part_size"]):
        suffix = f"{index // p['confirmation_part_size']:03d}"
        prep = f"confirm-prepare-{suffix}"; run = f"confirm-{suffix}"
        subset = parents[index:index + p["confirmation_part_size"]]
        add(prep, "confirm_prepare", "shared", ["freeze"], 1800 if full else 300 if smoke else 900,
            cpus=8, parent_ids=subset, splits=["confirmation"])
        add(run, "confirm", "shared", ["freeze", prep], 1800 if full else 600 if smoke else 1200,
            parent_ids=subset, prepared_unit=prep)
        confirmations.append(run)
    add("aggregate", "aggregate", "shared", ["freeze", *confirmations], 600 if full else 180)
    add("scaling", "scaling", "shared", ["freeze", "aggregate"], 1200 if full else 300 if smoke else 600)
    add("report", "report", "shared", list(units), 600 if full else 240)
    return units


def build_protocol(profile="smoke"):
    if profile not in PROFILES:
        raise ValueError("Portfolio profile must be smoke, development or full")
    p = copy.deepcopy(frontier_protocol(profile)); full = profile == "full"; smoke = profile == "smoke"
    # Fresh continuous fields, not merely new IDs on previously inspected phases.
    offset = {"smoke": 51000000, "development": 52000000, "full": 53000000}[profile]
    old_min = min(r["field_seed"] for r in p["parents"])
    for r in p["parents"]:
        seed = offset + r["field_seed"] - old_min
        r.update(seed=seed, field_seed=seed, parent_id=f"portfolio-{r['split']}-{seed}",
                 field_cluster=f"portfolio-field-{seed}", continuous_field_id=f"portfolio-field-{seed}")
        rng = random.Random(seed)
        for term in r["field_terms"]:
            term["phase"] = rng.uniform(0., 2 * math.pi)
    p.update(schema=SCHEMA, version=1, models=list(FAMILIES), trainable_models=list(TRAINABLE),
        seeds=[6100011, 6100021, 6100031] if full else [6100011],
        mechanisms={k: dict(name=v, stages=[], question=v) for k, v in HYPOTHESES.items()}, combinations={},
        endpoint_steps=[1, 2, 4, 8], evaluation_horizons=[.12, .24] if not smoke else [.06, .12],
        primary_schedules=[[.02, .04, .06], [.06, .04, .02], [.04, .08, .12], [.12, .08, .04]]
            if not smoke else [[.02, .04], [.04, .02], [.04, .08]],
        confirmation_part_size=2, confirmation_after_checkpoint_freeze=True,
        scientific_failure_blocks_all_research=False, automatic_budget_expansion=False)
    p["confirm_schedules"] = endpoint_schedules(p)
    p["validation_horizons"] = sorted(set(p["validation_horizons"] + p["evaluation_horizons"]))
    p["classical_frontier_steps"] = p["endpoint_steps"]
    p["training"].update(updates=200 if full else 2 if smoke else 24,
        tuning_updates=40 if full else 1 if smoke else 4,
        learning_rates=[.001, .0003] if not smoke else [.001],
        subset_sizes=[8,32] if full else [2] if smoke else [4,8],
        validation_every=25 if full else 1 if smoke else 6,
        optimizer="AdamW", trial_seconds=120 if full else 15 if smoke else 30,
        fit_iterations=100, peak_weight=.1,
        loss="dimensionless defect-normalized mean-square plus squared maximum error",
        non_neural_fit="training-only ridge least-squares initialization followed by bounded matched RMS-plus-peak objective; validation selects initialization",
        ridge_candidates=[1e-8, 1e-4] if not smoke else [1e-6],
        reference_noise_floor=True, stopping="bounded updates and per-unit walltime; retain selected initialization",
        optimizer_controls=[dict(id="default", clip_grad_norm=1., loss_scale=1.),
                            dict(id="unclipped_scaled", clip_grad_norm=None, loss_scale=.01)],
        fairness_views=["equal_data", "measured_training_compute", "equal_inference_cost", "validation_selected_frontier"])
    p["portfolio_budget"] = dict(scope="discretionary training and exploratory prototypes only; shared audit/reference/confirmation separately bounded",
        fractions=dict(A=.55, B=.25, C=.20),
        probe_seconds=dict(A=3960, B=1800, C=1440) if full else
                      dict(A=660, B=300, C=240) if smoke else dict(A=1650, B=750, C=600),
        exploration_protected=True, reallocation="requires an explicitly recorded new development protocol; no automatic transfer")
    p["hypothesis_thresholds"] = dict(
        representation_error_ratio=1.5, learned_error_ratio=1.1, practical_speedup=1.2,
        confidence_level=.95, maximum_coverage_regression=.05, required_feasible_fraction=.95,
        comparison_unit="independent field; schedules/grids/seeds paired within field",
        frozen_control="quad2_fixed", learned_comparators=["quad2_amplitude", "quad2_nodes", "quad2_joint", "quad2_linear"],
        multiple_comparisons="declared primary tests; exploratory subgroup estimates unadjusted and labeled",
        mathematical_proof=False, primary_interval_lower_bound=1.,
        benefit_requires="declared effect threshold and field-cluster interval lower bound above one; coverage noninferiority checked separately")
    p["exploration"] = dict(seed=offset+900000, prototype_ids=["X01", "X02", "X03"],
        X01=dict(parity_tolerance=1e-10, relative_rmse=.001, break_even_queries=32),
        X02=dict(resolved_twin_tolerance=1e-12, target_separation=1e-8, memory_rmse_ratio=.5),
        X03=dict(relative_error=.01, total_speedup=1.10),
        novelty="new to this project; no publication-priority claim",
        confirmation_access=False)
    p["diagnostics"] = dict(grid=8 if smoke else 16, horizon=.12,
        regimes=["low", "high_pair", "rough", "near_nyquist"],
        terms=["spatial", "temporal", "quadrature", "compression", "optimization", "reference", "implementation", "deployment"],
        profile_families=["quad2_fixed", "quad2_conditioned", "quad4_full", "df", "etdrk4", "fno_small"],
        repetitions=2 if smoke else 5, profiling_separate_from_timing=True)
    p["selection"] = dict(split="validation", primary_family="quad2_conditioned",
        frozen_before_confirmation=True, keep_all_declared_attribution_controls=True,
        selection_metric="joint RMS and maximum error with reference uncertainty; measured validation cost",
        test_truth_selects_deployment=False)
    p["policies"].update(model_family="quad2_conditioned", automatic_deployment=False,
        reason="No new learned deployment policy until an independently measured solver-cost margin exists")
    p["scaling"].update(families=["quad2_conditioned", "quad2_fixed", "fno_small", "fno_standard", "df", "etdrk4"],
        grids=[64,128,256] if full else [8,16] if smoke else [32,64],
        maximum_cases=6 if full else 2, reference_required_for_speed_claim=True,
        case_order="descending batch size, then ascending grid, then target track; omitted cells remain NA")
    p["claims"] = dict(representation="retained signed nonlinear interactions before projection",
        learning="improvement beyond matched normalized analytic and non-neural fitted controls",
        neural="credible locally adapted FNO on equal data and measured compute, not a paper reproduction",
        generalization="paired new fields/physics/grids/horizons within declared scope",
        solver="complete accuracy-qualified cost with identical schedule opportunities",
        numerical="a distilled analytic rule may be a successful result without neural inference",
        published_fno_reproduction=False, universal_worst_case=False)
    p["units"] = _units(p)
    p["stages"] = list(p["units"])
    p["gpu_stages"] = [k for k,v in p["units"].items() if v["device"] == "cuda"]
    p["budgets"] = {k:{q:v[q] for q in ("seconds", "cpus", "mem_gib", "walltime")} for k,v in p["units"].items()}
    # GPU test identity remains frozen and independent of collection order.
    gradients = [f for f in TRAINABLE if f not in ("quad2_amplitude", "quad2_linear")]
    p["gpu_test_cases"] = [f"test_portfolio_cuda_limits[{t}-{f}]" for t in p["tracks"] for f in FAMILIES] + [
        f"test_portfolio_cuda_gradients[{t}-{f}]" for t in p["tracks"] for f in gradients] + [
        "test_portfolio_cuda_fitted_control[quad2_amplitude]", "test_portfolio_cuda_fitted_control[quad2_linear]",
        "test_portfolio_cuda_visible_allocation"]
    return p


def plan_units(protocol):
    return list(protocol["units"].values())


def dependencies(protocol, unit):
    return list(protocol["units"][unit]["dependencies"])


def resource_for(protocol, unit):
    return {k:protocol["units"][unit][k] for k in ("cpus", "mem_gib", "walltime", "device", "seconds")}


def validate_protocol(protocol):
    if not isinstance(protocol,dict) or protocol.get("schema") != SCHEMA or protocol.get("profile") not in PROFILES:
        raise ValueError("Invalid portfolio protocol")
    if digest(protocol) != digest(build_protocol(protocol["profile"])):
        raise ValueError("Portfolio declaration changed; start a newly declared development protocol")
    seen=set()
    for key,u in protocol["units"].items():
        if key!=u["id"] or not set(u["dependencies"])<=seen:
            raise ValueError("Portfolio dependency graph is not a unique acyclic declared plan")
        hours,minutes,seconds=map(int,u["walltime"].split(":")); wall=3600*hours+60*minutes+seconds
        if not 0 < u["seconds"] < wall <=2700 or u["mem_gib"]*1024>110000 or u["mem_gib"]>48 or u["cpus"]>8:
            raise ValueError("Portfolio unit exceeds desktop bounds")
        if u["device"]=="cuda" and u["cpus"]!=4:
            raise ValueError("GPU allocation requires four CPUs")
        seen.add(key)
    seeds={}; ids=set()
    for r in protocol["parents"]:
        if r["parent_id"] in ids or r["field_seed"] in seeds:
            raise ValueError("Portfolio parent identities and phase seeds must be independent")
        ids.add(r["parent_id"]);seeds[r["field_seed"]]=r["split"]
    amounts=protocol["portfolio_budget"]["probe_seconds"];total=sum(amounts.values())
    if any(not math.isclose(amounts[k]/total, f) for k,f in protocol["portfolio_budget"]["fractions"].items()):
        raise ValueError("Protected exploration budget changed")
    return protocol
