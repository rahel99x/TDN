"""Frozen question coverage, controlled factors and desktop resource admission."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tdn.analysis.agenda.protocol import STAGES, QUESTION_STAGES, build_protocol, validate_protocol


@pytest.mark.parametrize("profile", ["smoke", "development", "full"])
def test_declared_budget_fits_scheduler_and_physical_cores(profile):
    protocol = build_protocol(profile)
    validate_protocol(protocol)
    assert protocol["hardware"]["scheduler_memory_mib"] == 110000
    for stage, values in protocol["budgets"].items():
        assert values["mem_gib"] * 1024 < 110000
        assert values["cpus"] <= (8 if stage in STAGES[:2] else 4)
        h, m, s = map(int, values["walltime"].split(":"))
        assert values["seconds"] + 120 < 3600 * h + 60 * m + s <= 2700
    assert protocol["gpu_memory"] == dict(soft_cap_gib=18, soft_fraction=.75, hard_device_used_fraction=.9)


def test_full_design_is_crossed_and_keeps_spectrum_phase_controlled():
    protocol = build_protocol("full")
    fresh = [p for p in protocol["parents"] if p["split"] == "confirmation"]
    assert len(fresh) == 32
    baseline = [p for p in fresh if p["physics_index"] == 0]
    assert len(baseline) == 8
    common = ("seed", "phase", "mean", "variance", "amplitude")
    assert len({tuple(p[k] for k in common) for p in baseline[:4]}) == 1
    assert {p["spectrum"] for p in baseline[:4]} == {"low", "rough", "high_pair", "near_nyquist"}
    for p in baseline[4:]:
        reference = baseline[1]
        assert sum(p[k] != reference[k] for k in ("phase", "mean", "variance", "amplitude", "spectrum")) == 1
    for index in range(8):
        rows = [p for p in fresh if p["controlled_field_index"] == index]
        assert {(p["kappa"], p["reaction_rate"]) for p in rows} == {(.001, .5), (.001, 8.), (.03, .5), (.03, 8.)}
        assert len({p["continuous_field_id"] for p in rows}) == 1
    assert protocol["continuum_parent_ids"] == [p["parent_id"] for p in baseline]


def test_development_never_reuses_full_parent_or_phase_seed():
    banks = {profile: build_protocol(profile)["parents"] for profile in ("smoke", "development", "full")}
    for first, second in (("smoke", "full"), ("development", "full"), ("smoke", "development")):
        for key in ("parent_id", "seed", "continuous_field_id"):
            assert {p[key] for p in banks[first]}.isdisjoint({p[key] for p in banks[second]})
    for parents in banks.values():
        by_split = {split: {p["seed"] for p in parents if p["split"] == split}
                    for split in ("train", "validation", "calibration", "confirmation")}
        for split, values in by_split.items():
            assert all(values.isdisjoint(other) for name, other in by_split.items() if name != split)


def test_confirmation_step_queries_are_unseen_and_distinct_from_horizon_identity():
    protocol = build_protocol("full")
    seen = set(protocol["train_horizons"] + protocol["validation_horizons"])
    assert all(h not in seen for schedule in protocol["confirm_schedules"] for h in schedule)
    assert protocol["confirm_schedules"][0] == list(reversed(protocol["confirm_schedules"][1]))
    assert len({round(sum(schedule), 12) for schedule in protocol["confirm_schedules"]}) == 2
    assert len({tuple(schedule) for schedule in protocol["confirm_schedules"]}) == 5


def test_all_literature_questions_have_executable_stage_coverage():
    requirements = json.loads((Path(__file__).resolve().parents[1] / "docs/AGENDA_REQUIREMENTS.json").read_text())
    assert set(QUESTION_STAGES) == {f"Q{i}" for i in range(1, 8)}
    assert all(stages and set(stages) <= set(STAGES) for stages in QUESTION_STAGES.values())
    raw = json.dumps(requirements)
    assert all(f'"Q{i}"' in raw for i in range(1, 8))
    assert set(build_protocol("full")["classical"]) == {
        "strang_reaction_first", "strang_diffusion_first", "etdrk4", "gl3_fused", "spectral_mean"}


def test_full_trial_and_checkpoint_selection_coverage():
    protocol = build_protocol("full")
    assert {stage: len(rows) for stage, rows in protocol["stage_trials"].items()} == {
        "controls": 11, "optimize": 36, "compression": 18, "kernel": 4}
    assert len({row["trial_id"] for rows in protocol["stage_trials"].values() for row in rows}) == 69
    slots = protocol["confirmation_selection"]
    assert len(slots) == protocol["maximum_confirmation_variants"] == 12
    compression = [s for s in slots if s["stage"] == "compression"]
    assert len(compression) == 9
    assert {s["seed"] for s in compression} == set(protocol["seeds"])
    assert {s["family"] for s in compression} == {"source", "precompress_source", "fno_source_matched"}


@pytest.mark.parametrize("path,value", [(("budgets", "prepare", "mem_gib"), 110),
    (("success_criteria", "minimum_overall_coverage"), .1), (("reference_tolerance",), .01),
    (("policy", "safety_factor"), 1.), (("maximum_confirmation_variants",), 1)])
def test_budget_or_scientific_plan_cannot_be_silently_relaxed(path, value):
    protocol = copy.deepcopy(build_protocol("full"))
    node = protocol
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match="immutable"):
        validate_protocol(protocol)


def test_controller_protocol_import_does_not_load_numerical_packages():
    root = Path(__file__).resolve().parents[1]
    script = "from tdn.analysis.agenda.protocol import build_protocol; import sys; build_protocol('full'); assert not {'torch','numpy','scipy'} & set(sys.modules)"
    response = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True)
    assert response.returncode == 0, response.stderr
