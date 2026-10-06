"""Complete rollout evidence, conservative frontiers and retained interruptions."""
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.work_precision import experiment
from tdn.analysis.work_precision.protocol import (
    DEFAULTS, METHODS, NSTEPS, TABLE_IDS, make_protocol, validate_config, validate_rows,
)
from tdn.numerics.types import Equation, Geometry


def declaration():
    config = validate_config(deepcopy(DEFAULTS), smoke=True)
    plans = experiment.plan(config)
    # One declared case keeps regression work bounded; its full scientific grid remains.
    case = plans["cases"][0]
    plans["cases"] = [case]
    plans["timing_orders"] = [item for item in plans["timing_orders"] if item["case_id"] == case["case_id"]]
    plans["expected_ids"] = {name: [key for key in ids if key.startswith(case["case_id"] + "/")]
                             for name, ids in plans["expected_ids"].items()}
    return make_protocol(config, plans, {}, [])


def empty_tables():
    return {name: [] for name in TABLE_IDS}


@pytest.fixture(scope="module")
def measured_case():
    protocol, tables = declaration(), empty_tables()
    with torch.no_grad():
        experiment._run_case(protocol, protocol["plans"]["cases"][0], tables, lambda: None)
    tables["frontier_rows"] = experiment.select_frontiers(protocol, tables["candidate_rows"], tables["reference_rows"])
    return protocol, tables


def test_real_case_retains_reference_parity_and_every_complete_candidate(measured_case):
    protocol, tables = measured_case
    validate_rows(tables, protocol["plans"], complete=True)
    assert len(tables["candidate_rows"]) == 6 * 7
    assert len(tables["frontier_rows"]) == 7 * 2 * 4
    assert len(tables["parity_rows"]) == 10
    ref = tables["reference_rows"][0]
    assert ref["reference_accepted"]
    assert ref["reference_4n"] == 2 * ref["reference_2n"] == 4 * ref["reference_n"]
    assert ref["uncertainty_max_estimate"] == pytest.approx(
        math.sqrt(math.prod(protocol["plans"]["cases"][0]["grid"])) * ref["uncertainty_rms"])
    assert all(row["status"] == "PASS" for row in tables["parity_rows"])
    for row in tables["candidate_rows"]:
        assert row["status"] == "VALID" and row["trajectory_completed"]
        assert row["completed_steps"] == row["nsteps"]
        assert row["error_rms"] >= 0 and row["error_max"] >= row["error_rms"]
        assert row["error_upper_rms"] == row["error_rms"] + ref["uncertainty_rms"]
        assert row["error_upper_max_estimate"] == row["error_max"] + ref["uncertainty_max_estimate"]


def test_five_raw_interleaved_repeats_keep_preparation_and_warmup_separate(measured_case):
    protocol, tables = measured_case
    for row in tables["candidate_rows"]:
        order = next(item for item in protocol["plans"]["timing_orders"] if item["nsteps"] == row["nsteps"])
        assert row["warmup"]["repeat_index"] == -1
        assert row["warmup"]["method_order_index"] == order["warmup_order"].index(row["variant"])
        repeats = row["timing_repeats"]
        assert len(repeats) == 5
        assert [item["repeat_index"] for item in repeats] == list(range(5))
        assert [item["method_order_index"] for item in repeats] == [
            variants.index(row["variant"]) for variants in order["measured_orders"]]
        seconds = [item["seconds"] for item in repeats]
        assert row["prepared_min_seconds"] == min(seconds) > 0
        assert row["prepared_median_seconds"] == statistics.median(seconds)
        assert row["prepared_max_seconds"] == max(seconds)
        assert row["preparation_seconds"] > 0
        assert row["setup_inclusive_median_seconds"] == row["preparation_seconds"] + statistics.median(seconds)
        assert all(item["work_counts_complete"] for item in repeats)
        assert all(item["work"] == row["work_per_rollout"] == row["warmup"]["work"] for item in repeats)


@pytest.mark.parametrize("variant,transforms", zip(METHODS, (2, 5, 9, 18, 26, 18, 3)))
def test_prepared_work_accumulates_across_actual_steps(variant, transforms):
    spec = experiment.cases()[0]
    u = experiment.state(spec)
    prepared = experiment.prepare_step(u, .01, Equation(spec["kappa"], spec["reaction_rate"]),
        Geometry(tuple(spec["grid"]), tuple(spec["lengths"])), variant)
    single, _, error = experiment._rollout(prepared, u, 1, lambda: None, 0, 0)
    assert error is None and single["status"] == "VALID"
    multiple, _, error = experiment._rollout(prepared, u, 3, lambda: None, 1, 0)
    assert error is None and multiple["status"] == "VALID"
    assert multiple["work"] == {key: 3 * count for key, count in single["work"].items()}
    assert multiple["work"]["fft_forward"] + multiple["work"]["fft_inverse"] == 3 * transforms


@pytest.mark.parametrize("invalid", [float("nan"), 2.])
def test_invalid_rollout_stops_at_first_invalid_state_with_attempted_work(invalid):
    u = experiment.state(experiment.cases()[0])
    def step(value, *, work):
        work["fft_forward"] = work.get("fft_forward", 0) + 1
        return torch.full_like(value, invalid)
    sample, _, error = experiment._rollout(step, u, 4, lambda: None, 2, 3)
    assert error is None
    assert sample["status"] == "INVALID" and sample["completed_steps"] == 1
    assert not sample["trajectory_completed"] and sample["work_counts_complete"]
    assert sample["work"] == {"fft_forward": 1}
    assert sample["seconds"] > 0


@pytest.mark.parametrize("minimum,status,steps", [(-1.8e-14, "INVALID", 1), (-1e-14, "VALID", 4)])
def test_rollout_bounds_match_prepared_step_roundoff_allowance(minimum, status, steps):
    u = experiment.state(experiment.cases()[0])
    calls = []
    def step(value, *, work):
        calls.append(None)
        if len(calls) > 1 and minimum < -64 * torch.finfo(value.dtype).eps:
            pytest.fail("An invalid endpoint was passed to the next prepared step")
        return torch.full_like(value, minimum)
    sample, _, error = experiment._rollout(step, u, 4, lambda: None, 0, 0)
    assert error is None and sample["status"] == status
    assert sample["completed_steps"] == len(calls) == steps


@pytest.mark.parametrize("error,status", [(RuntimeError("kernel failed"), "FAILED"),
    (experiment.ScreenIncomplete("budget"), "INCOMPLETE"),
    (InterruptedError("signal"), "INCOMPLETE"), (KeyboardInterrupt(), "INCOMPLETE")])
def test_failed_rollout_returns_original_exception_and_partial_work(error, status):
    calls = []
    def step(value, *, work):
        work["fft_forward"] = work.get("fft_forward", 0) + 1
        calls.append(None)
        if len(calls) == 2:
            raise error
        return value
    u = experiment.state(experiment.cases()[0])
    sample, _, returned = experiment._rollout(step, u, 4, lambda: None, 0, 0)
    assert returned is error and sample["status"] == status
    assert sample["completed_steps"] == 1 and not sample["trajectory_completed"]
    assert sample["work"] == {"fft_forward": 2} and not sample["work_counts_complete"]
    assert sample["failure_reason"].startswith(type(error).__name__)


def stub_teacher(u, spec, equation, geometry, config, check):
    check()
    return SimpleNamespace(state=u.clone()), {
        "reference_id": spec["case_id"] + "/reference", "case_id": spec["case_id"],
        "reference_accepted": True, "uncertainty_rms": 1e-10, "uncertainty_max_estimate": 1e-9,
    }


def test_partial_invalid_candidate_has_null_final_horizon_errors(monkeypatch):
    protocol, tables = declaration(), empty_tables()
    real_prepare = experiment.prepare_step
    def prepare(u, h, equation, geometry, variant):
        if variant != "strang":
            return real_prepare(u, h, equation, geometry, variant)
        def step(value, *, work):
            work["fft_forward"] = work.get("fft_forward", 0) + 1
            return value + 2
        step.metadata = {}
        return step
    monkeypatch.setattr(experiment, "prepare_step", prepare)
    monkeypatch.setattr(experiment, "_teacher", stub_teacher)
    experiment._run_case(protocol, protocol["plans"]["cases"][0], tables, lambda: None)
    rows = [row for row in tables["candidate_rows"] if row["variant"] == "strang" and row["nsteps"] > 1]
    assert len(rows) == 5
    for row in rows:
        assert row["status"] == "INVALID" and not row["trajectory_completed"]
        assert row["completed_steps"] == 1 and len(row["timing_repeats"]) == 5
        for key in ("error_rms", "error_max", "error_mean", "error_upper_rms", "error_upper_max_estimate",
                    "prepared_median_seconds", "setup_inclusive_median_seconds"):
            assert row[key] is None
        assert all(sample["work"]["fft_forward"] == 1 for sample in row["timing_repeats"])
    validate_rows(tables, protocol["plans"], complete=False)


def frontier_data():
    protocol = declaration()
    spec = protocol["plans"]["cases"][0]
    references = [{"case_id": spec["case_id"], "reference_accepted": True,
                   "uncertainty_rms": 1e-7, "uncertainty_max_estimate": 2e-7}]
    values = [(1, .001, .002, 0., 0.), (2, .2, .21, 1.9995e-4, .003),
              (4, .3, .35, 4e-5, 4e-4), (8, .4, .405, 2e-5, 2e-5),
              (16, .5, .505, 3e-6, 3e-6), (32, .31, .311, 1e-4, 1e-4)]
    candidates = [{"candidate_id": f"{spec['case_id']}/strang/n{n}", "case_id": spec["case_id"],
                   "variant": "strang", "nsteps": n, "status": "INVALID" if n == 1 else "VALID",
                   "trajectory_completed": n != 1, "reference_accepted": True,
                   "error_rms": rms, "error_max": maximum,
                   "error_upper_rms": rms + 1e-7, "error_upper_max_estimate": maximum + 2e-7,
                   "prepared_median_seconds": seconds, "setup_inclusive_median_seconds": setup}
                  for n, seconds, setup, rms, maximum in values]
    return protocol, candidates, references


def test_posthoc_frontiers_use_adjusted_errors_and_separate_norm_and_setup_winners():
    protocol, candidates, references = frontier_data()
    rows = experiment.select_frontiers(protocol, candidates, references)
    assert len(rows) == 7 * 2 * 4
    indexed = {(row["variant"], row["norm"], row["tolerance"]): row for row in rows}
    rms = indexed["strang", "rms", 2e-4]
    maximum = indexed["strang", "max", 2e-4]
    assert rms["status"] == maximum["status"] == "FEASIBLE"
    assert rms["selected_nsteps"] == 4 and maximum["selected_nsteps"] == 32
    assert rms["setup_selected_nsteps"] == 32
    assert rms["adjusted_error"] == pytest.approx(4.01e-5)
    assert rms["feasible_candidate_count"] == 4
    assert indexed["strang", "rms", 2e-3]["selected_nsteps"] == 2
    assert indexed["strang", "rms", 2e-6]["status"] == "NO_FEASIBLE_CANDIDATE"
    assert all(row["selected_nsteps"] != 1 for row in rows)
    assert all(row["status"] == "INCONCLUSIVE" for row in rows if row["variant"] != "strang")


@pytest.mark.parametrize("damage", ["rejected_reference", "missing_reference", "missing_candidate",
                                    "incomplete_candidate", "failed_candidate"])
def test_unresolved_reference_or_grid_cannot_produce_a_feasible_frontier(damage):
    protocol, candidates, references = frontier_data()
    if damage == "rejected_reference":
        references[0]["reference_accepted"] = False
    elif damage == "missing_reference":
        references.clear()
    elif damage == "missing_candidate":
        candidates.pop()
    else:
        candidates[-1]["status"] = "INCOMPLETE" if damage == "incomplete_candidate" else "FAILED"
    rows = experiment.select_frontiers(protocol, candidates, references)
    assert {row["status"] for row in rows} == {"INCONCLUSIVE"}
    assert all(row["selected_candidate_id"] is None for row in rows)


@pytest.mark.parametrize("reason", ["stop", "budget", "unexpected"])
def test_interrupted_or_failed_run_persists_partial_rows_without_a_completion_seal(tmp_path, monkeypatch, reason):
    protocol = declaration()
    stop, clock = SimpleNamespace(requested=False), [0.]
    def partial(protocol, spec, tables, check):
        _, row = stub_teacher(experiment.state(spec), spec, None, None, None, check)
        tables["reference_rows"].append(row)
        if reason == "unexpected":
            raise RuntimeError("deliberate unexpected failure")
        if reason == "stop":
            stop.requested = True
        else:
            clock[0] = 181.
        check()
    monkeypatch.setattr(experiment, "_run_case", partial)
    summary = experiment.run(protocol, tmp_path, stop=stop, clock=lambda: clock[0])
    assert summary["status"] == ("FAILED" if reason == "unexpected" else "INCOMPLETE")
    assert summary["scientific_outcome"] == "INCONCLUSIVE"
    assert summary["reported_references"] == 1 and summary["reported_candidates"] == 0
    canonical = json.loads((tmp_path / "work-precision.json").read_text())
    assert len(canonical["reference_rows"]) == 1
    assert len(canonical["unreported_ids"]["candidate_rows"]) == 42
    assert canonical["errors"] and not (tmp_path / "COMPLETED").exists()
    assert all(row["status"] == "INCONCLUSIVE" for row in canonical["frontier_rows"])


def test_unexpected_method_failure_retains_failed_candidate_and_attempted_cost(tmp_path, monkeypatch):
    protocol = declaration()
    def prepare(*args):
        def broken(value, *, work):
            work["fft_forward"] = 1
            raise RuntimeError("deliberate kernel failure")
        broken.metadata = {}
        return broken
    monkeypatch.setattr(experiment, "_teacher", stub_teacher)
    monkeypatch.setattr(experiment, "prepare_step", prepare)
    summary = experiment.run(protocol, tmp_path)
    assert summary["status"] == "FAILED" and summary["scientific_outcome"] == "INCONCLUSIVE"
    canonical = json.loads((tmp_path / "work-precision.json").read_text())
    failed = [row for row in canonical["candidate_rows"] if row["status"] == "FAILED"]
    assert len(failed) == 1
    row = failed[0]
    assert not row["trajectory_completed"] and row["error_rms"] is None and row["error_max"] is None
    assert row["warmup"]["seconds"] > 0 and row["preparation_seconds"] > 0
    assert row["warmup"]["work"] == {"fft_forward": 1} and not row["warmup"]["work_counts_complete"]
    assert "deliberate kernel failure" in row["failure_reason"]
    assert not (tmp_path / "COMPLETED").exists()


@pytest.mark.parametrize("reason", ["stop", "budget"])
def test_stop_mid_rollout_retains_partial_candidate_without_endpoint_errors(tmp_path, monkeypatch, reason):
    protocol = declaration()
    stop, clock = SimpleNamespace(requested=False), [0.]
    trigger_h = protocol["plans"]["cases"][0]["final_time"] / 2
    def prepare(u, h, equation, geometry, variant):
        def step(value, *, work):
            work["fft_forward"] = work.get("fft_forward", 0) + 1
            if h == trigger_h:
                if reason == "stop":
                    stop.requested = True
                else:
                    clock[0] = 181.
            return value
        step.metadata = {}
        return step
    monkeypatch.setattr(experiment, "_teacher", stub_teacher)
    monkeypatch.setattr(experiment, "prepare_step", prepare)
    summary = experiment.run(protocol, tmp_path, stop=stop, clock=lambda: clock[0])
    assert summary["status"] == "INCOMPLETE"
    canonical = json.loads((tmp_path / "work-precision.json").read_text())
    partial = [row for row in canonical["candidate_rows"] if row["nsteps"] == 2 and row["warmup"]]
    assert len(partial) == 1
    row = partial[0]
    assert row["status"] == "INCOMPLETE" and row["completed_steps"] == 1
    assert not row["trajectory_completed"] and row["warmup"]["work"] == {"fft_forward": 1}
    assert row["warmup"]["seconds"] > 0
    assert all(row[key] is None for key in ("error_rms", "error_max", "error_mean",
                                          "error_upper_rms", "error_upper_max_estimate"))
    assert not (tmp_path / "COMPLETED").exists()


def test_cli_incomplete_run_maps_to_interrupted_tower_and_keeps_exit_75(tmp_path, monkeypatch):
    import yaml
    from tdn import cli, reporting
    from tdn.runtime import metadata, precision, preflight, storage

    script = Path(__file__).resolve().parents[1] / "scripts" / "work_precision.py"
    spec = importlib.util.spec_from_file_location("work_precision_cli_interrupt_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(DEFAULTS))
    run_dir = tmp_path / "run"
    monkeypatch.setattr(storage, "configure_storage", lambda: None)
    monkeypatch.setattr(storage, "contained_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(preflight, "verify_runtime", lambda *args: None)
    monkeypatch.setattr(preflight, "execution_mode", lambda: "local")
    monkeypatch.setattr(metadata, "software_metadata", lambda: {"source_tree_sha256": "fixture"})
    monkeypatch.setattr(precision, "reference_precision", lambda: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setenv("TDN_TOWER_DIR", str(tmp_path / "previous-report"))
    monkeypatch.setattr(reporting, "attach_report", lambda *args, **kwargs: (tmp_path / "report", True))
    captured = {}
    def finalize(report, **kwargs):
        captured.update(kwargs)
        return True
    monkeypatch.setattr(cli, "finalize_tower_report", finalize)
    def incomplete(protocol, directory, **kwargs):
        result = {"status": "INCOMPLETE", "computational_status": "INCOMPLETE",
                  "scientific_outcome": "INCONCLUSIVE"}
        for name in ("summary.json", "work-precision.json"):
            (directory / name).write_text(json.dumps(result))
        (directory / "summary.txt").write_text("Interrupted scientific run\n")
        return result
    monkeypatch.setattr(experiment, "run", incomplete)
    assert module.main(["--config", str(config), "--run-dir", str(run_dir), "--smoke"]) == 75
    assert captured["state"] == "INTERRUPTED" and captured["exit_code"] == 75
    assert captured["metadata"]["scientific_status"] == "INCOMPLETE"
    assert captured["metadata"]["actually_ran"]
    for name in ("stage.json", "summary.json", "work-precision.json"):
        assert json.loads((run_dir / name).read_text())["status"] == "INCOMPLETE"
    assert not (run_dir / "COMPLETED").exists() and not (run_dir / "manifest.json").exists()
    assert os.environ["TDN_TOWER_DIR"] == str(tmp_path / "previous-report")


@pytest.mark.parametrize("changed", ["source", "config"])
def test_cli_fingerprint_mismatch_preserves_compact_unsealed_evidence(tmp_path, monkeypatch, changed):
    import yaml
    from tdn import cli, reporting
    from tdn.runtime import metadata, precision, preflight, storage

    script = Path(__file__).resolve().parents[1] / "scripts" / "work_precision.py"
    spec = importlib.util.spec_from_file_location("work_precision_cli_fingerprint_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(DEFAULTS))
    run_dir = tmp_path / "run"
    generation = [0]
    monkeypatch.setattr(storage, "configure_storage", lambda: None)
    monkeypatch.setattr(storage, "contained_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(preflight, "verify_runtime", lambda *args: None)
    monkeypatch.setattr(preflight, "execution_mode", lambda: "local")
    monkeypatch.setattr(metadata, "software_metadata", lambda: {
        "source_tree_sha256": f"source-{generation[0]}"})
    monkeypatch.setattr(precision, "reference_precision", lambda: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(reporting, "attach_report", lambda *args, **kwargs: (tmp_path / "report", True))
    finalized = {}
    def finalize(report, **kwargs):
        finalized.update(kwargs)
        return True
    monkeypatch.setattr(cli, "finalize_tower_report", finalize)
    repeats = [{"repeat_index": index, "seconds": seconds, "status": "VALID",
                "trajectory_completed": True, "completed_steps": 4,
                "failure_reason": None, "work": {"fft_forward": 4, "fft_inverse": 4}}
               for index, seconds in enumerate((.0000123456789, .0000111, .0000108, .0000112, .0000115))]
    candidate = {"candidate_id": "fixed/strang/n4", "timing_repeats": repeats,
                 "error_rms": 1.2e-7, "error_max": 3.4e-7}
    def completed(protocol, directory, **kwargs):
        result = {"status": "COMPLETED", "computational_status": "COMPLETED",
                  "scientific_outcome": "OBSERVED_MIXED"}
        metadata.write_json(directory / "summary.json", result)
        experiment.write_canonical(directory / "work-precision.json", {
            **result, "candidate_rows": [candidate], "reference_rows": [],
            "parity_rows": [], "frontier_rows": [], "errors": []})
        (directory / "summary.txt").write_text("Computational COMPLETED before sealing\n")
        if changed == "source":
            generation[0] += 1
        else:
            config.write_text(config.read_text() + "\n# configuration changed during execution\n")
        return result
    monkeypatch.setattr(experiment, "run", completed)
    assert module.main(["--config", str(config), "--run-dir", str(run_dir), "--smoke"]) == 1
    assert finalized["state"] == "FAILED" and finalized["exit_code"] == 1
    stage = json.loads((run_dir / "stage.json").read_text())
    assert stage["status"] == "FAILED"
    expected_reason = "Execution source changed" if changed == "source" else "Configuration changed"
    assert expected_reason in stage["error"]
    raw = (run_dir / "work-precision.json").read_text()
    assert raw.endswith("\n") and raw.count("\n") == 1
    canonical = json.loads(raw)
    assert canonical["candidate_rows"] == [candidate]
    assert canonical["candidate_rows"][0]["timing_repeats"] == repeats
    for payload in (canonical, json.loads((run_dir / "summary.json").read_text())):
        assert payload["status"] == payload["computational_status"] == "INCOMPLETE"
        assert payload["scientific_outcome"] == "INCONCLUSIVE"
        assert expected_reason in payload["sealing_error"]
    assert not (run_dir / "COMPLETED").exists() and not (run_dir / "manifest.json").exists()
