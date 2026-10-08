"""Frozen independent cohorts, fair comparison inventory and hardware ceilings."""
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tdn.analysis.frontier.protocol import (
    GATES, GPU_STAGES, GPU_TEST_CASES, MODEL_IDS, STAGES, TRACKS, TRAINABLE,
    build_protocol, validate_protocol,
)
from tdn.research.protocol import digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("profile", ["smoke", "development", "full"])
def test_five_questions_keep_the_complete_separate_stage_chain(profile):
    p = build_protocol(profile)
    assert validate_protocol(p) is p
    assert p["stages"] == list(STAGES)
    assert p["gpu_stages"] == list(GPU_STAGES)
    assert tuple(p["mechanisms"]) == ("G1", "G2", "G3", "G4", "G5")
    assert p["classical_frontier_steps"] == [1, 2, 4, 8]
    assert set().union(*(set(g["stages"]) for g in p["mechanisms"].values())) == set(STAGES) - {"report"}
    assert all(g["question"].endswith("?") for g in p["mechanisms"].values())
    assert digest(p) == digest(build_protocol(profile))


def test_returned_declarations_do_not_alias_the_canonical_gate_registry():
    p = build_protocol("smoke")
    original = deepcopy(GATES)
    # Preserve the global registry even if this test catches a shared-reference
    # bug, avoiding contamination of independent tests in the same process.
    try:
        p["mechanisms"]["G1"]["question"] = "Silently change the scientific question?"
        assert GATES == original
        with pytest.raises(ValueError, match="frozen declaration"):
            validate_protocol(p)
    finally:
        GATES.clear()
        GATES.update(original)


def test_fields_coefficients_seeds_and_profiles_are_independent():
    profile_ids, profile_seeds, profile_fields = set(), set(), set()
    for profile in ("smoke", "development", "full"):
        p = build_protocol(profile)
        ids, seeds, fields, ownership = set(), set(), set(), {}
        for parent in p["parents"]:
            assert parent["parent_id"] not in ids
            ids.add(parent["parent_id"])
            assert parent["field_seed"] not in seeds
            seeds.add(parent["field_seed"])
            assert parent["field_cluster"] == parent["continuous_field_id"]
            assert ownership.setdefault(parent["field_cluster"], parent["split"]) == parent["split"]
            coefficients = digest({key: parent[key] for key in ("field_terms", "mean", "variance")})
            assert coefficients not in fields
            fields.add(coefficients)
            for term in parent["field_terms"]:
                assert len(term["mode"]) == 2 and all(type(k) is int for k in term["mode"])
                assert all(abs(k) < p["train_grid"] / 2 for k in term["mode"])
                assert 0 <= term["phase"] < 2 * math.pi
                assert math.isfinite(term["weight"]) and term["weight"] > 0
        assert not ids & profile_ids and not seeds & profile_seeds and not fields & profile_fields
        profile_ids.update(ids); profile_seeds.update(seeds); profile_fields.update(fields)


@pytest.mark.parametrize("profile,counts", [
    ("smoke", dict(development=2, train=2, validation=2, calibration=2, confirmation=2, scaling=2, policy=2)),
    ("development", dict(development=6, train=8, validation=4, calibration=4, confirmation=8, scaling=2, policy=4)),
    ("full", dict(development=12, train=32, validation=16, calibration=16, confirmation=24, scaling=4, policy=12)),
])
def test_all_seven_split_quotas_are_field_counts_not_endpoint_counts(profile, counts):
    p = build_protocol(profile)
    assert Counter(row["split"] for row in p["parents"]) == counts
    assert len(p["parents"]) == sum(counts.values())
    assert len({row["field_cluster"] for row in p["parents"]}) == sum(counts.values())
    assert p["independent_unit"] == "continuous_field_cluster"
    assert p["confirmation_after_checkpoint_freeze"] is True
    assert p["policies"]["calibration_split"] == "calibration"


def test_normal_and_stress_fields_share_both_spatial_target_tracks():
    p = build_protocol("full")
    assert tuple(p["tracks"]) == TRACKS == ("discrete", "continuum")
    expected = {"low", "mixed", "high_pair", "rough", "near_nyquist", "boundary_mean", "reaction_stiff", "diffusion_stiff"}
    for split in ("development", "train", "validation", "calibration", "confirmation", "policy"):
        rows = [r for r in p["parents"] if r["split"] == split]
        assert {r["regime"] for r in rows} == expected
        assert {r["category"] for r in rows} == {"favorable", "typical", "stress"}
    # No target-specific favorable-only subset is authorized by the protocol.
    assert "continuum_parent_ids" not in p
    assert p["screening"]["tracks"] == p["tracks"]
    assert p["claims"]["universal_worst_case"] is False


def test_coefficient_shift_labels_mean_actually_unseen_physics():
    p = build_protocol("full")
    training = {(r["kappa"], r["reaction_rate"]) for r in p["parents"] if r["split"] == "train"}
    shifted = [r for r in p["parents"] if r["distribution"] == "coefficient_shift"]
    ordinary = [r for r in p["parents"] if r["split"] == "confirmation" and r not in shifted]
    assert len(shifted) == 8 and len(ordinary) == 16
    assert all(r["split"] == "confirmation" and (r["kappa"], r["reaction_rate"]) not in training for r in shifted)
    assert all((r["kappa"], r["reaction_rate"]) in training for r in ordinary)
    assert {r["regime"] for r in shifted} == {r["regime"] for r in ordinary}


def test_model_inventory_keeps_analytic_and_frozen_attribution_controls():
    from tdn.analysis.frontier.models import FAMILIES, TRAINABLE_FAMILIES
    from tdn.analysis.frontier.neural import expected_model_specs
    p = build_protocol("full")
    assert p["models"] == list(MODEL_IDS) == list(FAMILIES)
    assert p["trainable_models"] == list(TRAINABLE) == list(TRAINABLE_FAMILIES)
    assert {"rank1", "rank1_frozen", "rank1_postcompression", "analytic_quad", "analytic_quad_cubic",
            "fno_small", "fno_standard", "direct_fno", "df", "etdrk4"} == set(p["models"])
    rows = expected_model_specs(p)
    assert len(rows) == len({r["model_id"] for r in rows}) == 80
    for family in ("rank1", "rank1_frozen", "rank1_postcompression", "fno_small", "fno_standard", "direct_fno"):
        cells = {(r["track"], r["train_count"], r["seed"]) for r in rows if r["family"] == family}
        assert cells == {(t, n, s) for t in TRACKS for n in (8, 32) for s in p["seeds"]}
    assert p["model_configs"]["fno_standard"]["width"] > p["model_config"]["width"]
    assert p["model_configs"]["fno_standard"]["depth"] > p["model_config"]["depth"]
    assert p["claims"]["published_fno_reproduction"] is False


@pytest.mark.parametrize("profile,finals,tunings,updates,examples", [
    ("smoke", 20, 10, 30, 30), ("development", 32, 20, 760, 760), ("full", 80, 20, 12800, 51200),
])
def test_trial_update_and_example_budgets_include_tuning(profile, finals, tunings, updates, examples):
    from tdn.analysis.frontier.neural import expected_model_specs
    p = build_protocol(profile)
    records = expected_model_specs(p)
    trainable_records = [r for r in records if r["family"] in TRAINABLE]
    tuning_count = len(TRACKS) * len(TRAINABLE) * len(p["training"]["learning_rates"])
    total = tuning_count * p["training"]["tuning_updates"] + len(trainable_records) * p["training"]["updates"]
    assert len(records) == finals and tuning_count == tunings
    assert total == updates and total * p["training"]["batch_size"] == examples
    assert p["training"]["initialization_control"] is True
    assert "validation only" in p["training"]["candidate_selection"]


@pytest.mark.parametrize("profile", ["smoke", "development", "full"])
def test_allocations_fit_node_and_leave_time_for_reporting(profile):
    p = build_protocol(profile)
    for stage, budget in p["budgets"].items():
        assert 4 <= budget["cpus"] <= (4 if stage in GPU_STAGES else 8)
        assert budget["mem_gib"] in (32, 48)
        assert budget["mem_gib"] * 1024 <= 110000
        h, m, s = map(int, budget["walltime"].split(":"))
        wall = 3600 * h + 60 * m + s
        assert 0 < budget["seconds"] < wall <= 2700
    assert p["automatic_budget_expansion"] is False
    assert p["scientific_failure_blocks_all_research"] is False
    assert not any("pending" in key for key in p)
    assert p["state_precision"] == "float32" and p["teacher_precision"] == "float64" and p["tf32"] is False
    from tdn.runtime.desktop_slurm import desktop_memory_policy
    assert desktop_memory_policy() == {"soft_cap_bytes": 18 * 2**30, "soft_fraction": .75, "hard_fraction": .9}


@pytest.mark.parametrize("mutation", ["gate", "memory", "seed", "field", "targets", "models", "updates", "profile", "risk", "pending"])
def test_mutating_preregistered_science_or_limits_is_rejected(mutation):
    p = deepcopy(build_protocol("smoke"))
    if mutation == "gate": p["mechanisms"]["G1"]["question"] = "Other question?"
    elif mutation == "memory": p["budgets"]["train"]["mem_gib"] = 64
    elif mutation == "seed": p["parents"][0]["field_seed"] += 1
    elif mutation == "field": p["parents"][0]["field_terms"][0]["phase"] += .01
    elif mutation == "targets": p["targets"][0] *= 10
    elif mutation == "models": p["models"].remove("fno_standard")
    elif mutation == "updates": p["training"]["updates"] += 1
    elif mutation == "profile": p["profile"] = "full"
    elif mutation == "risk": p["policies"]["alpha"] = .5
    else: p["pending_job_cap"] = 1
    with pytest.raises(ValueError, match="frozen declaration"):
        validate_protocol(p)


def test_fixed_horizons_reverse_order_and_targets_are_distinct():
    p = build_protocol("full")
    schedules = p["confirm_schedules"]
    assert schedules[0] == list(reversed(schedules[1]))
    assert schedules[3] == list(reversed(schedules[4]))
    assert {round(sum(s), 12) for s in schedules} == {.12, .24}
    assert p["evaluation_horizons"] == [.12, .24]
    assert p["targets"] == [2e-4, 2e-5, 2e-6]
    assert p["practical_speedup"] == 1.2
    assert p["scaling"]["reference_required_for_speed_claim"] is True
    assert p["policies"]["estimated_cost_margin_required"] is True


def test_native_gpu_inventory_matches_actual_pytest_collection():
    assert len(GPU_TEST_CASES) == len(set(GPU_TEST_CASES)) == 45
    env = {key: value for key, value in os.environ.items() if key not in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "TDN_REQUIRE_GPU_TESTS")}
    result = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/test_frontier_gpu.py",
        "-o", "cache_dir=.runtime/frontier-collection-cache"], cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    collected = [line.split("::", 1)[1] for line in result.stdout.splitlines() if line.startswith("tests/test_frontier_gpu.py::")]
    assert collected == list(GPU_TEST_CASES)


def test_bootstrap_protocol_import_does_not_require_numerical_dependencies():
    code = "import sys; sys.path.insert(0,sys.argv[1]); from tdn.analysis.frontier.protocol import build_protocol,validate_protocol; p=build_protocol('full'); validate_protocol(p); assert not {'torch','numpy','scipy'} & set(sys.modules); print(len(p['parents']))"
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(ROOT)], cwd=ROOT,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "116"
