"""Frozen P1 independent-parent/source scaling contract, separate from R7/R8."""
from __future__ import annotations

import copy
import json
import math

from .r7_protocol import VERIFY_TESTS as R7_VERIFY_TESTS

SCHEMA = "tdn.response_preconditioner.scaling.v1"
FAMILIES = ("smooth_spectral", "localized_periodic", "periodic_transition", "multiscale_discrete")
SPLITS = ("train", "validation", "diagnostic")
VERIFY_TESTS = (*R7_VERIFY_TESTS, *(
    f"tests/test_response_preconditioner_scaling_{name}.py"
    for name in ("data", "analysis", "workflow")), *(
    "tests/test_response_preconditioner_scaling_launch.py::" + name for name in (
        "test_shell_contract_and_non_native_rejection",
        "test_cli_study_rejects_standalone_and_gpu_before_science",
        "test_recipe_has_only_scaling_science_and_cpu_memory",
        "test_scaling_rejects_resource_expansion",
        "test_prepare_separate_frozen_protocol_and_modes",
        "test_recursive_artifacts_preserve_binary_and_reject_symlinks",
        "test_stage_preserves_failed_outputs_and_rejects_promotion",
        "test_metrics_keep_cohorts_families_and_unresolved_rows",
        "test_preflight_failure_never_starts_study_and_is_preserved",
        "test_export_terminal_snapshot_and_refusals")))


def protocol(command="study"):
    if command not in ("verify", "study"):
        raise ValueError("P1 scaling supports verify and study only")
    return dict(
        schema=SCHEMA, phase="P1", command=command, device="cpu", target="fd_same_grid",
        decision="BOUNDED_DEVELOPMENT", scientific_promotion=False,
        cohort="new development parents; one nested balanced cohort/order; not fresh confirmation",
        parent_seeds={"train": list(range(2026101601, 2026101633)),
                      "validation": list(range(2026101701, 2026101709)),
                      "diagnostic": list(range(2026101801, 2026101809))},
        parent_families=list(FAMILIES),
        parent_family_rule="interleave four families by zero-based parent_index modulo four in each split",
        parent_counts=[8, 16, 32], calibration_counts=[2, 4, 8],
        parent_generator="scaling.data.v1: disjoint per-parent morphology, amplitude/mean, and physics streams",
        parent_seed_rule="SeedSequence([parent_seed,101/102/103]) for morphology/state scale/physics",
        parent_ranges=dict(mean=[.25, .65], amplitude=[.04, .18], kappa=[.009, .017], reaction_rate=[1.7, 2.9]),
        parent_field_rule="center discrete morphology and normalize its maximum absolute value; u=mean+amplitude*shape; round once to FP32",
        parent_identity="SHA256 of actual contiguous little-endian FP32 field plus canonical grid/length/FP64 physics metadata; seed and family stored separately",
        morphology_scope="periodic 5x5 discrete morphology coverage only; no continuum-resolved sharp-interface claim",
        grid=[5, 5], lengths=[1., 1.],
        intervals=[[0., .04], [0., .08], [.02, .04], [.02, .06], [.04, .06], [.04, .08]],
        anchor="original smooth df AnchorPath(u0) at absolute time, never restarted at s",
        calibration_sources=8, withheld_sources=4,
        calibration_families=["dense_random"]*8,
        withheld_families=["dense_random", "low_frequency", "high_frequency", "constant"],
        source_seed_rule="SeedSequence([parent_seed,context_index,701/702]) calibration/withheld streams, independent of parent construction",
        source_definition="unit physical RMS; dense normal, cos(2pi*(x+y)+phase), cos(4pi*(x+y)+phase), ones",
        acquisition="train calibration eight plus withheld four per context; validation and diagnostic withheld four only",
        source_prefix_rule="first 2/4/8 entries of one fixed eight-source calibration stream; never regenerate per cell",
        model_precision="FP64 from actual FP32 initial states", model="unchanged R7 five branches and 40-weight fixed feature readout",
        mass="uniform physical cell mass / domain volume; RMS squared is mean squares",
        ridge_menu=[1e-10, 1e-6], zero_candidate=True,
        training="original action-only positive ridge; equal parent/context/source/cell mass; train calibration prefixes only",
        selection="per scaling cell minimum validation-withheld physical action MSE; initialization first wins exact ties",
        freeze="seal all nine selected readouts before any train-withheld or diagnostic truth acquisition",
        learning_curve_scope="nine paired cells, one nested parent order; finite-cohort trend, not a population learning-law estimate",
        reference=dict(tolerance=1e-10, initial_steps=4, max_steps=256,
                       max_seconds_per_query=3., independent_tolerance=1e-12,
                       independent_max_rhs_calls=4096),
        reference_count=2688, max_reference_queries=2688,
        reference_query_definition="one U/G source/context pair with refined Lawson and independent DOP853 on the identical anchor",
        reference_caps="at most 1016 Lawson steps and 8192 independent RHS calls per query with a shared three-second limit; total 900-second deadline can stop earlier",
        reference_resolution_multiple=10.,
        ordered_panels=[2, 8], lawson_steps=[2, 4, 8],
        expected_fit_cells=9, expected_withheld_queries=1152, expected_metric_rows=16128,
        timing=dict(rounds=5, warmup=1, seed=2026101608, threads=1,
                    parent_indices=[0, 1, 2, 3], selected_parent_count=32, selected_source_count=8,
                    scope="largest 32-parent/eight-source cell only; first diagnostic parent of each family, all six contexts, four withheld sources; every source rebuilds setup, no cross-source caching",
                    cold="first measured call after accuracy acquisition; not cold process startup",
                    order="five randomized paired rounds after first and warmup, raw min/median/max; no confidence claim",
                    memory="actual worker process-tree sampled RSS, not method-specific peak",
                    expected_bundles=1008, max_complete_source_calls=4032),
        parent_summary="per-parent paired errors, median/worst and improved-parent fractions; eight diagnostic parents with two per family, no bootstrap confidence interval or confirmation claim",
        budget=dict(study_seconds=900., native_cpus=4, native_memory_gib=8,
                    native_wall_minutes=20, native_only=True, automatic_expansion=False),
        utility_gate="NOT_DEFINED: descriptive accuracy/cost and data scaling, no speedup or automatic promotion",
        historical_gate="B STOP unchanged",
        deferred=["additional parent cohorts", "active optimal design", "defect-source learning", "new regularizer", "architecture changes", "dense prediction sensitivity maps", "local oracles", "full operator audit", "analytic coefficient control", "rollout", "GPU", "fresh confirmation"],
        verify_tests=list(VERIFY_TESTS),
    )


def planned_counts(config):
    """Exact intended denominators, including queries not acquired on failure."""
    contexts = len(config["intervals"])
    counts = {split: len(config["parent_seeds"][split]) for split in SPLITS}
    references = contexts*(counts["train"]*config["calibration_sources"]
                           + sum(counts.values())*config["withheld_sources"])
    withheld = contexts*sum(counts.values())*config["withheld_sources"]
    cells = len(config["parent_counts"])*len(config["calibration_counts"])
    methods = cells + len(config["ordered_panels"]) + len(config["lawson_steps"])
    bundles = len(config["timing"]["parent_indices"])*contexts*6*(
        1 + config["timing"]["warmup"] + config["timing"]["rounds"])
    return dict(reference_count=references, expected_fit_cells=cells,
                expected_withheld_queries=withheld, expected_metric_rows=withheld*methods,
                expected_timing_bundles=bundles,
                max_complete_source_calls=bundles*config["withheld_sources"])


def _portable(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_config(config, *, allow_fixture=False):
    """Reject changed production work; private reduced fixtures need opt-in.

    Neither preparation nor public execution may opt into fixture acceptance.
    Fixtures can only reduce acquisition/fit/timing populations and budgets;
    the operator, reference policy, physical generator, and objective stay fixed.
    """
    if not isinstance(config, dict):
        raise ValueError("Scaling protocol must be an object")
    canonical = protocol(config.get("command"))
    if config.get("_implementation_fixture") is not True:
        if _portable(config) != _portable(canonical):
            raise ValueError("Changed frozen P1 scaling protocol; prepare the exact production contract")
    else:
        if not allow_fixture:
            raise ValueError("Private implementation fixture is forbidden in production")
        mutable = {"_implementation_fixture", "cohort", "parent_seeds", "parent_counts",
                   "calibration_counts", "calibration_sources", "withheld_sources",
                   "calibration_families", "withheld_families", "intervals",
                   "reference_count", "max_reference_queries", "expected_fit_cells",
                   "expected_withheld_queries", "expected_metric_rows", "timing", "budget"}
        if set(config) != set(canonical) | {"_implementation_fixture"}:
            raise ValueError("Unknown or missing fixture protocol keys")
        for key in set(canonical)-mutable:
            if _portable(config[key]) != _portable(canonical[key]):
                raise ValueError(f"Fixture changed protected protocol field: {key}")
        for split in SPLITS:
            seeds = config["parent_seeds"].get(split, [])
            if not seeds or seeds != canonical["parent_seeds"][split][:len(seeds)]:
                raise ValueError("Fixture parent seeds must be nonempty production prefixes")
        for key, maximum in (("calibration_sources", 8), ("withheld_sources", 4)):
            if type(config[key]) is not int or not 1 <= config[key] <= maximum:
                raise ValueError("Fixture source counts must be reduced positive integers")
            role = key.removesuffix("_sources")
            if config[role+"_families"] != canonical[role+"_families"][:config[key]]:
                raise ValueError("Fixture source families must be original prefixes")
        for key, limit in (("parent_counts", len(config["parent_seeds"]["train"])),
                           ("calibration_counts", config["calibration_sources"])):
            levels = config[key]
            if (not isinstance(levels, list) or not levels
                    or any(type(x) is not int or not 1 <= x <= limit for x in levels)
                    or levels != sorted(set(levels)) or levels[-1] != limit):
                raise ValueError("Fixture scaling counts must increase to the acquired cap")
        if (not config["intervals"] or any(x not in canonical["intervals"] for x in config["intervals"])
                or len({tuple(x) for x in config["intervals"]}) != len(config["intervals"])):
            raise ValueError("Fixture intervals must be a unique production subset")
        budget = config["budget"]
        if set(budget) != set(canonical["budget"]) or any(
                budget[k] != canonical["budget"][k] for k in budget if k != "study_seconds"):
            raise ValueError("Fixture cannot expand or change native resources")
        if (type(budget["study_seconds"]) not in (int, float)
                or not math.isfinite(budget["study_seconds"])
                or not 0 < budget["study_seconds"] <= 900):
            raise ValueError("Fixture budget must be positive and at most 900 seconds")
        timing, base_timing = config["timing"], canonical["timing"]
        if set(timing) != set(base_timing):
            raise ValueError("Fixture timing keys must match production")
        for key in set(timing)-{"rounds", "warmup", "parent_indices", "selected_parent_count", "selected_source_count", "expected_bundles", "max_complete_source_calls"}:
            if timing[key] != base_timing[key]:
                raise ValueError("Fixture cannot change timing semantics")
        for key, upper, lower in (("rounds", 5, 1), ("warmup", 1, 0)):
            if type(timing[key]) is not int or not lower <= timing[key] <= upper:
                raise ValueError("Fixture timing repetitions exceed frozen limits")
        indices = timing["parent_indices"]
        if (not isinstance(indices, list) or not indices or len(indices) != len(set(indices))
                or any(type(x) is not int or x < 0 or x >= min(4, len(config["parent_seeds"]["diagnostic"])) for x in indices)):
            raise ValueError("Fixture timing parents must be available original indices")
        if (timing["selected_parent_count"] != config["parent_counts"][-1]
                or timing["selected_source_count"] != config["calibration_counts"][-1]):
            raise ValueError("Timing must use the largest acquired scaling cell")
    seeds = [seed for split in SPLITS for seed in config["parent_seeds"][split]]
    if any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Parent seeds must be distinct nonnegative integers across splits")
    counts = planned_counts(config)
    if any(config[key] != counts[key] for key in (
            "reference_count", "expected_fit_cells", "expected_withheld_queries", "expected_metric_rows")):
        raise ValueError("Frozen acquisition/evaluation denominators do not match the planned workload")
    if config["max_reference_queries"] != counts["reference_count"]:
        raise ValueError("Planned queries must equal the frozen reference cap")
    if (config["timing"]["expected_bundles"] != counts["expected_timing_bundles"]
            or config["timing"]["max_complete_source_calls"] != counts["max_complete_source_calls"]):
        raise ValueError("Timing denominators do not match the frozen workload")
    return counts


def _fixture_config():
    """48-query private test fixture; never a scientifically eligible study."""
    config = copy.deepcopy(protocol())
    config.update(_implementation_fixture=True, cohort="48-query private implementation fixture only",
                  parent_seeds={split: values[:4 if split == "train" else 2]
                                for split, values in config["parent_seeds"].items()},
                  parent_counts=[2, 4], calibration_counts=[2, 4],
                  calibration_sources=4, calibration_families=["dense_random"]*4,
                  intervals=[[.02, .04]])
    config["budget"]["study_seconds"] = 45.
    config["timing"].update(rounds=1, warmup=0, parent_indices=[0],
                            selected_parent_count=4, selected_source_count=4)
    counts = planned_counts(config)
    for key in ("reference_count", "expected_fit_cells", "expected_withheld_queries", "expected_metric_rows"):
        config[key] = counts[key]
    config["max_reference_queries"] = counts["reference_count"]
    config["timing"]["expected_bundles"] = counts["expected_timing_bundles"]
    config["timing"]["max_complete_source_calls"] = counts["max_complete_source_calls"]
    validate_config(config, allow_fixture=True)
    return config
