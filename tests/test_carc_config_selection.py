"""Config-selection policy tests perform no training or scheduler submission."""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from tdn.config import load_config
from tdn.data.generation import _generation_configuration
from tdn.runtime import preflight, storage

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("select_carc_config", ROOT / "scripts/select_carc_config.py")
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


@pytest.fixture
def base_config():
    return load_config(ROOT / "configs/smoke.yaml")


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    shutil.copyfile(ROOT / "configs/smoke.yaml", root / "configs/smoke.yaml")
    calls = []
    monkeypatch.setattr(selector, "configure_storage", lambda: root)
    monkeypatch.setattr(selector, "contained_path", lambda path: storage.contained_path(path, root=root))
    monkeypatch.setattr(selector, "verify_runtime", lambda device, stage: calls.append((device, stage)))
    monkeypatch.setattr(selector, "execution_mode", lambda: "unit-policy-mock")
    return root, calls


def test_plan_keeps_common_immutable_teacher_design_and_fixed_budget(base_config):
    plan = selector.make_plan(base_config)
    designs = [_generation_configuration(record["config"]) for record in plan["candidates"].values()]
    assert len(plan["candidates"]) == 4
    assert plan["repetitions"] == 3 and plan["steps_per_trial"] == 12
    assert all(design == _generation_configuration(base_config) for design in designs)
    assert plan["diagnostic_parents_used_for_selection"] is False
    assert plan["confirmatory_parents_opened"] is False


@pytest.mark.parametrize("section,key,value", [
    ("problem", "grid", [16, 16]),
    ("data", "validation_count", 4),
    ("training", "max_steps", 13),
    ("validation", "tolerance", .003),
    ("validation", "require_headroom", True),
    ("precision", "network_autocast", "bfloat16"),
    ("problem", "kappa_range", [.002, .006]),
])
def test_selection_refuses_expanded_or_changed_scientific_design(base_config, section, key, value):
    base_config[section][key] = value
    with pytest.raises(ValueError, match="Bounded config selection requires"):
        selector.make_plan(base_config)


def test_prepare_checks_runtime_and_records_plan_before_any_training(isolated_project):
    root, calls = isolated_project
    destination, _config, plan = selector.prepare_run(root / "selection")
    assert calls == [("cpu", "config-selection")]
    assert destination == root / "selection"
    assert json.loads((destination / "plan.json").read_text())["base_config_sha256"] == plan["base_config_sha256"]
    assert not (destination / "dataset").exists()
    assert not (destination / "summary.json").exists()


def test_actual_carc_runtime_policy_refuses_login_without_run_outputs(isolated_project, monkeypatch):
    root, _calls = isolated_project
    for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT", "TDN_EXECUTION_MODE", "CONDA_PREFIX"):
        monkeypatch.delenv(name, raising=False)
    # Test the real policy against an isolated project marked as the CARC root.
    # No Slurm variables are fabricated and no numerical work executes.
    monkeypatch.setattr(preflight, "CARC_ROOT", root)
    monkeypatch.setattr(preflight, "project_root", lambda: root)
    monkeypatch.setattr(preflight.sys, "prefix", str(root / ".venv"))
    monkeypatch.setattr(preflight.getpass, "getuser", lambda: "aadaniel")
    monkeypatch.setattr(selector, "verify_runtime", preflight.verify_runtime)
    with pytest.raises(ValueError, match="srun in an allocation"):
        selector.prepare_run(root / "selection")
    assert not (root / "selection").exists()


def test_existing_run_is_preserved(isolated_project):
    root, _calls = isolated_project
    destination = root / "selection"
    destination.mkdir()
    evidence = destination / "preserved.txt"
    evidence.write_text("retain prior work")
    with pytest.raises(ValueError, match="already exists"):
        selector.prepare_run(destination)
    assert evidence.read_text() == "retain prior work"
    assert not (destination / "plan.json").exists()


def test_outside_run_path_refused_before_outputs(isolated_project):
    root, _calls = isolated_project
    destination = root.parent / "outside-selection"
    with pytest.raises(ValueError, match="escapes project root"):
        selector.prepare_run(destination)
    assert not destination.exists()


def test_symlink_escape_refused_before_outputs(isolated_project):
    root, _calls = isolated_project
    outside = root.parent / "outside"
    outside.mkdir()
    link = root / "redirect"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Account cannot create directory symlinks")
    with pytest.raises(ValueError, match="escapes project root"):
        selector.prepare_run(link / "selection")
    assert not (outside / "selection").exists()


@pytest.mark.parametrize("which", ["source", "config", "selector"])
def test_changed_inputs_stop_before_trials(isolated_project, monkeypatch, which):
    root, _calls = isolated_project
    destination, _config, plan = selector.prepare_run(root / "selection")
    if which == "source":
        provenance = copy.deepcopy(plan["source_provenance"])
        provenance["python_source_hash"] = "changed"
        monkeypatch.setattr(selector, "source_provenance", lambda: provenance)
    elif which == "config":
        path = Path(plan["base_config_path"])
        path.write_text(path.read_text() + "\n# changed while running\n")
    else:
        plan["selector_sha256"] = "different"
    with pytest.raises(RuntimeError, match="changed during config selection"):
        selector.verify_unchanged(plan)
    assert not (destination / "measurements.json").exists()
    assert not (destination / "summary.json").exists()


def trial(mean_error, *, feasible=True, status="COMPLETE", steps=12):
    return {"status": status, "steps": steps,
            "best_validation": {"feasible": feasible, "mean_error": mean_error}}


def test_selection_is_validation_only_and_rejects_infeasible_checkpoint():
    candidates = {
        "validation_best": {"trials": [trial(.0002)] * 3, "median_wall_seconds": 2.,
                            "diagnostic_error": .1},
        "diagnostic_best": {"trials": [trial(.0003)] * 3, "median_wall_seconds": 1.,
                            "diagnostic_error": 0.},
        "infeasible": {"trials": [trial(.0001, feasible=False)] * 3, "median_wall_seconds": .1},
    }
    assert selector.select_winner(candidates) == "validation_best"


def test_interrupted_or_incomplete_trial_cannot_select_prior_feasible_checkpoint():
    candidate = {"trials": [trial(.0001), trial(.0001), trial(.0001, status="PAUSED_BUDGET", steps=10)],
                 "median_wall_seconds": 1.}
    assert selector.select_winner({"partial": candidate}) is None
    candidate["trials"].pop()
    with pytest.raises(ValueError, match="all three"):
        selector.select_winner({"partial": candidate})
