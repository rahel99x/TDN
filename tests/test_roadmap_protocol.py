"""Frozen mechanism coverage, independent cohorts and desktop resource limits."""
import copy

import pytest

from tdn.analysis.roadmap.protocol import (
    COMBINATION_STAGES, GPU_STAGES, GPU_TEST_CASES, MECHANISM_STAGES,
    MODEL_IDS, STAGES, build_protocol, validate_protocol,
)
from tdn.research.protocol import digest


@pytest.mark.parametrize("profile", ["smoke", "development", "full"])
def test_all_mechanisms_combinations_and_declared_models_are_frozen(profile):
    p = build_protocol(profile)
    assert validate_protocol(p) is p
    assert set(p["mechanisms"]) == {f"M{i:02}" for i in range(24)}
    assert set(p["combinations"]) == {f"C{i}" for i in range(5)}
    assert p["models"] == list(MODEL_IDS)
    assert len(set(p["models"])) == len(p["models"])
    for identity, mechanism in p["mechanisms"].items():
        assert mechanism["stages"] == MECHANISM_STAGES[identity]
        assert set(mechanism["stages"]) <= set(STAGES)
        assert mechanism["gaps"]
    for identity, combination in p["combinations"].items():
        assert combination["stages"] == COMBINATION_STAGES[identity]
        assert set(combination["mechanisms"]) <= set(p["mechanisms"])
    assert digest(p) == digest(build_protocol(profile))


def test_cohorts_and_profiles_do_not_reuse_independent_field_clusters():
    seen_profiles = set()
    for profile in ("smoke", "development", "full"):
        p = build_protocol(profile)
        cohorts = {}
        for parent in p["parents"]:
            assert cohorts.setdefault(parent["field_cluster"], parent["split"]) == parent["split"]
        assert not seen_profiles.intersection(cohorts)
        seen_profiles.update(cohorts)
        assert len({v["parent_id"] for v in p["parents"]}) == len(p["parents"])
    full = build_protocol("full")
    confirmation = [p for p in full["parents"] if p["split"] == "confirmation"]
    assert len(confirmation) == 48
    assert len({p["field_cluster"] for p in confirmation}) == 24
    for cluster in {p["field_cluster"] for p in confirmation}:
        members = [p for p in confirmation if p["field_cluster"] == cluster]
        assert members[0]["field_terms"] == members[1]["field_terms"]
        assert (members[0]["kappa"], members[0]["reaction_rate"]) != (members[1]["kappa"], members[1]["reaction_rate"])


@pytest.mark.parametrize("profile", ["smoke", "development", "full"])
def test_jobs_fit_actual_110000_mib_cpu_and_4090_caps(profile):
    p = build_protocol(profile)
    assert p["hardware"]["scheduler_memory_mib"] == 110000
    assert p["hardware"]["dedicated_vram_gib"] == 24
    for stage, budget in p["budgets"].items():
        assert 1 <= budget["cpus"] <= (4 if stage in GPU_STAGES else 8)
        assert 0 < budget["mem_gib"] <= 48
        assert budget["mem_gib"] * 1024 < 110000
        h, m, s = map(int, budget["walltime"].split(":"))
        wall = 3600 * h + 60 * m + s
        assert 0 < budget["seconds"] < wall <= 45 * 60
    assert p["gpu_memory"] == {"soft_cap_gib": 18, "soft_fraction": 0.75, "hard_device_used_fraction": 0.9}
    assert p["pending_job_cap"] is None
    assert p["automatic_budget_expansion"] is False
    assert p["scientific_failure_blocks_other_mechanisms"] is False


def test_gpu_readiness_registry_covers_every_model_without_skipping():
    p = build_protocol("full")
    assert p["gpu_test_cases"] == list(GPU_TEST_CASES)
    assert len(GPU_TEST_CASES) == 2 * len(MODEL_IDS) + 6
    for model in MODEL_IDS:
        assert f"test_roadmap_cuda_model_limits[{model}]" in GPU_TEST_CASES
        assert f"test_roadmap_cuda_model_gradient[{model}]" in GPU_TEST_CASES
    assert "test_roadmap_cuda_allocation_visible" in GPU_TEST_CASES


@pytest.mark.parametrize("mutation", ["omit_mechanism", "raise_memory", "change_parent", "alter_target", "omit_model", "alter_score"])
def test_protocol_changes_are_rejected_before_experiment(mutation):
    p = copy.deepcopy(build_protocol("smoke"))
    if mutation == "omit_mechanism":
        p["mechanisms"].pop("M23")
    elif mutation == "raise_memory":
        p["budgets"]["train"]["mem_gib"] = 120
    elif mutation == "change_parent":
        p["parents"][0]["field_terms"][0]["phase"] += 0.001
    elif mutation == "alter_target":
        p["targets"][0] *= 10
    elif mutation == "omit_model":
        p["models"].pop()
    else:
        p["scoring"]["mathematical_proof_claim"] = True
    with pytest.raises(ValueError, match="frozen declaration"):
        validate_protocol(p)


def test_validation_only_selection_and_trajectory_design_remain_explicit():
    p = build_protocol("full")
    assert p["data"]["confirmation_after_checkpoint_freeze"]
    assert p["training"]["preserve_initialization"]
    assert p["training"]["orientation"] == "diffusion_first"
    assert p["confirm_schedules"][0] == list(reversed(p["confirm_schedules"][1]))
    assert sum(p["confirm_schedules"][-1]) > sum(p["confirm_schedules"][0])
    assert len(p["seeds"]) == 3
    assert p["scoring"]["mathematical_proof_claim"] is False
    assert p["scoring"]["minimum_mechanism_categories"] == ["math", "gap"]


def test_continuum_subset_preserves_all_stress_regimes_and_paired_physics():
    from collections import Counter
    p = build_protocol("full")
    chosen = set(p["continuum_parent_ids"])
    confirmation = [v for v in p["parents"] if v["split"] == "confirmation"]
    subset = [v for v in confirmation if v["parent_id"] in chosen]
    assert len(subset) == 24
    assert len({v["field_cluster"] for v in subset}) == 12
    assert set(v["spectrum"] for v in subset) == set(v["spectrum"] for v in confirmation)
    assert set(Counter(v["spectrum"] for v in subset).values()) == {4}
    assert set(Counter(v["field_cluster"] for v in subset).values()) == {2}
    first_clusters = sorted({v["field_cluster"] for v in confirmation})[:p["policies"]["max_confirmation_clusters"]]
    assert all(v["parent_id"] in chosen for v in confirmation if v["field_cluster"] in first_clusters)
