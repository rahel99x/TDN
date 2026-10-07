"""Agenda publication preserves provenance, failures and bounded native reads."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tdn import agenda_reporting as reporting
from tdn.reporting import begin_report, finish_report
from tdn.tower_analytics import publish_outputs

ROOT = Path(__file__).resolve().parents[1]


def put(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def fixture(tmp_path, stage="confirm", rows=None, status="COMPLETED"):
    source = tmp_path / stage
    source.mkdir()
    report = Path(begin_report(source, name="TDN/agenda/" + stage, script="scripts/tower.py",
                              parameters={"benchmark_suite": "agenda"}, report_parent=tmp_path / "tower"))
    rows = [] if rows is None else rows
    put(source / "rows.json", {"schema": reporting.SCHEMA, "stage": stage, "rows": rows})
    put(source / "summary.json", {"schema": "tdn.agenda-summary/v1", "stage": stage,
        "status": status, "scientific_outcome": "INCONCLUSIVE", "counts": {"rows": len(rows)}})
    if status == "COMPLETED":
        seal(source)
    return source, report


def seal(source):
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()
             if path.is_file() and path.name not in {"stage.json", "manifest.json", "COMPLETED"}}
    put(source / "manifest.json", {"schema": "tdn.agenda-stage/v1", "files": files})
    (source / "COMPLETED").write_text(hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest() + "\n")


def table(report, name):
    with (report / "outputs" / (name + ".csv")).open(newline="") as stream:
        return list(csv.DictReader(stream))


def resolve(report, row, key="source_record"):
    value = json.loads((report / row["source_path"]).read_text())
    for item in row[key].split("/")[1:]:
        item = item.replace("~1", "/").replace("~0", "~")
        value = value[int(item)] if isinstance(value, list) else value[item]
    return value


def test_all_questions_preserve_measurements_failures_and_exact_provenance(tmp_path):
    examples = [
        {"record_type": "case", "row_id": "structure", "status": "OBSERVED", "oracle_spans": {"gate": {"rms": .03}}, "quadratic_error": .03},
        {"record_type": "reference", "row_id": "teacher", "reference_accepted": False, "reason": "unresolved"},
        {"record_type": "training", "row_id": "training", "checkpoint_selection": "SELECTED_INITIALIZATION", "training_loss": None, "gradient_by_regime": {"rough": .7}},
        {"record_type": "optimization", "row_id": "optimizer", "gradient_norm": 7., "correction_rms": .004},
        {"record_type": "capacity", "row_id": "capacity", "parameter_count": 192, "physical_gate_bypass": True, "fft_counts": {"forward": 2}},
        {"record_type": "kernel", "row_id": "rank", "rank": 4, "rank_gate_passed": False, "rank_errors": [.2, .1]},
        {"record_type": "confirmation", "row_id": "fresh", "track": "continuum", "error_rms": .001, "spatial_regression": True, "timing_seconds": .01, "checkpoint_selection": "SELECTED_INITIALIZATION"},
        {"record_type": "policy", "row_id": "policy", "false_accept": True, "false_accept_definitely_resolved": False, "rejected_seconds": .002, "fallback_seconds": .01, "defect_jvps": 4, "attempt_timings": [.001, .002]},
        {"record_type": "comparison", "row_id": "paired", "trained_pair_eligible": False, "ours_selection": "SELECTED_INITIALIZATION"},
    ]
    for row in examples:
        row["question_ids"] = ["Q1", "Q7"]
    source, report = fixture(tmp_path, rows=examples)
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    result = reporting.publish_agenda_outputs(report, [source, source])
    assert result["reporting_complete"]
    assert not result["application_completion_is_scientific_success"]
    assert result["sources"][0]["seal_status"] == "PROJECTED_CANONICAL_FILES_VERIFIED"
    assert sum(result["table_rows"].values()) == len(examples)
    for name in reporting.COLUMNS:
        rows = table(report, name)
        assert len(rows) == 1
        assert resolve(report, rows[0]) == examples[list(reporting.COLUMNS).index(name)]
        assert resolve(report, rows[0], "question_ids_record") == ["Q1", "Q7"]
        assert rows[0]["source_sha256"] == hashlib.sha256((source / "rows.json").read_bytes()).hexdigest()
    assert table(report, "agenda_training")[0]["training_loss"] == "null"
    assert table(report, "agenda_policy")[0]["false_accept"] == "true"
    assert table(report, "agenda_policy")[0]["false_accept_definitely_resolved"] == "false"
    assert table(report, "agenda_policy")[0]["defect_jvps"] == "4"
    assert table(report, "agenda_confirmation")[0]["error_rms"] == "0.001"
    assert table(report, "agenda_confirmation")[0]["checkpoint_selection"] == "SELECTED_INITIALIZATION"
    assert original == {path.name: path.read_bytes() for path in source.iterdir()}


def test_delegate_hashes_and_explicit_junit_do_not_import_neighbor(tmp_path):
    source, report = fixture(tmp_path, rows=[{"row_id": "candidate", "record_type": "confirmation", "status": "FAILED"}])
    tests = tmp_path / "reporter-tests/confirm"
    tests.mkdir(parents=True)
    (tests / "tests.xml").write_text('<testsuite name="allocated" tests="1" failures="0"><testcase name="real_test"/></testsuite>')
    neighbor = tmp_path / "reporter-tests/unrelated"
    neighbor.mkdir()
    (neighbor / "tests.xml").write_text('<testsuite name="neighbor" tests="1" failures="1"><testcase name="unrelated"><failure>failed</failure></testcase></testsuite>')
    result = publish_outputs(report, [source, tests])
    assert result["agenda"]["reporting_complete"]
    assert len(table(report, "tests")) == 1
    assert "real_test" in json.dumps(table(report, "tests"))
    assert "unrelated" not in json.dumps(table(report, "tests"))
    artifact = json.loads((report / "outputs/artifacts.json").read_text())
    entry = next(row for row in artifact["artifacts"] if row["path"].endswith("/rows.json"))
    assert entry["hash_status"] == "computed_agenda_projection"
    assert entry["projection"] == "outputs/agenda-tables.json"


def test_outer_manifest_object_hashes_remain_declared_for_unread_binary(tmp_path):
    source, report = fixture(tmp_path, rows=[{"record_type": "case", "row_id": "one"}])
    binary = source / "large.bin"
    binary.write_bytes(b"x" * ((1 << 20) + 1))
    seal(source)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["files"] = {name: {"sha256": digest, "bytes": (source / name).stat().st_size}
                         for name, digest in manifest["files"].items()}
    put(source / "manifest.json", manifest)
    (source / "COMPLETED").write_text(hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest() + "\n")
    result = publish_outputs(report, [source])
    assert result["agenda"]["reporting_complete"]
    artifacts = json.loads((report / "outputs/artifacts.json").read_text())["artifacts"]
    row = next(item for item in artifacts if item["path"].endswith("/large.bin"))
    assert row["hash_status"] == "declared_unverified"
    assert row["sha256"] == manifest["files"]["large.bin"]["sha256"]


@pytest.mark.parametrize("malformed", ["nonfinite", "duplicate", "count", "tamper", "unsealed", "missing_summary", "missing_type", "duplicate_id"])
def test_incomplete_or_corrupt_evidence_never_claims_complete_publication(tmp_path, malformed):
    source, report = fixture(tmp_path, rows=[{"row_id": "one", "record_type": "case", "status": "FAIL"}])
    if malformed == "nonfinite":
        (source / "rows.json").write_text('{"schema":"tdn.agenda-rows/v1","stage":"confirm","rows":[{"rms":NaN}]}')
    elif malformed == "duplicate":
        (source / "rows.json").write_text('{"schema":"tdn.agenda-rows/v1","stage":"confirm","rows":[],"rows":[]}')
    elif malformed == "count":
        value = json.loads((source / "summary.json").read_text()); value["counts"]["rows"] = 8
        put(source / "summary.json", value); seal(source)
    elif malformed == "tamper":
        value = json.loads((source / "rows.json").read_text()); value["rows"][0]["status"] = "PASS"
        put(source / "rows.json", value)
    elif malformed == "unsealed":
        (source / "COMPLETED").unlink()
    elif malformed == "missing_summary":
        (source / "summary.json").unlink()
    elif malformed == "missing_type":
        value = json.loads((source / "rows.json").read_text()); del value["rows"][0]["record_type"]
        put(source / "rows.json", value); seal(source)
    elif malformed == "duplicate_id":
        value = json.loads((source / "rows.json").read_text()); value["rows"] *= 2
        put(source / "rows.json", value); seal(source)
    result = reporting.publish_agenda_outputs(report, [source])
    assert not result["reporting_complete"]
    assert result["reporting_omission_count"] > 0
    assert not result["application_completion_is_scientific_success"]


def test_budget_omission_retains_raw_records_and_is_explicit(tmp_path, monkeypatch):
    source, report = fixture(tmp_path, rows=[{"record_type": "case", "row_id": str(i)} for i in range(3)])
    monkeypatch.setitem(reporting.TABLE_LIMITS, "agenda_cases", {"rows": 1, "bytes": 16 << 20})
    result = reporting.publish_agenda_outputs(report, [source])
    assert not result["reporting_complete"]
    assert result["table_rows"]["agenda_cases"] == 1
    assert len(json.loads((source / "rows.json").read_text())["rows"]) == 3


def test_aggregate_table_budget_is_enforced_before_publication(tmp_path, monkeypatch):
    source, report = fixture(tmp_path, rows=[{"record_type": "case", "row_id": "large", "reason": "x" * 1000},
        {"record_type": "policy", "row_id": "fallback", "reason": "x" * 1000}])
    headers = sum(reporting._header_bytes(columns) for columns in reporting.COLUMNS.values())
    monkeypatch.setattr(reporting, "MAX_OUTPUT_BYTES", headers + 100)
    result = reporting.publish_agenda_outputs(report, [source])
    assert not result["reporting_complete"]
    assert sum(result["table_rows"].values()) == 0
    assert sum(path.stat().st_size for path in (report / "outputs").glob("agenda_*.csv")) <= headers + 100


def test_legacy_sources_do_not_create_agenda_outputs(tmp_path):
    source = tmp_path / "audit"; source.mkdir()
    report = Path(begin_report(source, name="TDN/legacy", script="scripts/tower.py", report_parent=tmp_path / "tower"))
    assert reporting.publish_agenda_outputs(report, [source]) is None
    assert not (report / "outputs/agenda-results.json").exists()


def test_native_agenda_validation_uses_named_fixed_budget_and_preserves_consistency(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("agenda_tower_tools", ROOT / "scripts/tower.py")
    tools = importlib.util.module_from_spec(spec); spec.loader.exec_module(tools)
    source, report = fixture(tmp_path)
    monkeypatch.setattr(tools, "tower_profile", lambda root, report: ("fedora", root / ".tower/fedora-slurm.json"))
    monkeypatch.setattr(tools.shutil, "which", lambda _: sys.executable)
    monkeypatch.setattr(tools, "tower_interpreter", lambda _: (sys.executable, None))
    command, settings = tools.native_validation_command(ROOT, report)
    assert command[-4:] == ["--max-bytes", str(64 << 20), "--suite", "agenda"]
    assert settings["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    manifest = json.loads((report / "run.json").read_text())
    manifest["name"] = "TDN/consistency/neural"; manifest["parameters"] = {"benchmark_suite": "consistency"}
    put(report / "run.json", manifest)
    command, _ = tools.native_validation_command(ROOT, report)
    assert command[-2:] == ["--max-bytes", str(32 << 20)]
    assert "--suite" not in command


def test_complete_declared_confirmation_size_with_unmodified_native_tower(tmp_path):
    native = ROOT / ".runtime/tower-current"
    if not (native / "tower/artifacts.py").is_file():
        pytest.skip("Unmodified upstream Tower checkout is an optional native integration fixture")
    from tdn.analysis.agenda.protocol import build_protocol
    from tdn.runtime.metadata import write_json
    protocol = build_protocol("full")
    trials = [row for row in protocol["stage_trials"]["compression"]
              if row["family"] in ("source", "precompress_source", "fno_source_matched")]
    trials += [next(row for row in protocol["stage_trials"]["controls"] if row["family"] == family)
               for family in ("source_time", "source_closure")]
    trials += [protocol["stage_trials"]["kernel"][0]]
    methods = [(family, family, None) for family in protocol["classical"]] + [
        (trial["trial_id"], trial["family"], trial["seed"]) for trial in trials]
    candidates, frontiers = [], []
    for parent in [p for p in protocol["parents"] if p["split"] == "confirmation"]:
        tracks = ["discrete"] + (["continuum"] if parent["parent_id"] in protocol["continuum_parent_ids"] else [])
        for n in protocol["grids"]:
            for identifier, family, seed in methods:
                for track in tracks:
                    for index, schedule in enumerate(protocol["confirm_schedules"]):
                        row = {"record_type": "confirmation", "question_ids": ["Q1", "Q2", "Q4", "Q6", "Q7"],
                            "parent_id": parent["parent_id"], "continuous_field_id": parent["continuous_field_id"],
                            "trial_id": identifier, "family": family, "seed": seed, "grid_size": n,
                            "schedule": schedule, "schedule_index": index, "horizon": round(sum(schedule), 12),
                            "factors": {key: parent.get(key) for key in ("mean", "variance", "amplitude", "spectrum", "phase", "kappa", "reaction_rate")},
                            "checkpoint_selection": "TRAINED_CHECKPOINT" if seed is not None else "CLASSICAL", "device": "cuda",
                            "base_orientation": "reaction-first", "checkpoint_freeze_sha256": "a" * 64, "status": "COMPLETED",
                            "timing_seconds": .0012345678901234, "timing_samples": [.0012345678901234] * 3,
                            "complete_rollout": True, "preparation_included": True, "steps": len(schedule), "track": track,
                            "spatial_regression": False}
                        for key in ("error_rms", "error_max", "mean_error", "signed_mean_error", "spatial_rms",
                                    "uncertainty_rms", "uncertainty_max", "upper_rms", "upper_max", "base_error_rms",
                                    "base_error_max", "base_mean_error", "base_spatial_rms", "error_rms_vs_base",
                                    "error_max_vs_base", "mean_error_vs_base", "spatial_rms_vs_base"):
                            row[key] = .00000012345678901234
                        candidates.append(row)
                    for horizon in sorted({round(sum(schedule), 12) for schedule in protocol["confirm_schedules"]}):
                        for norm in protocol["norms"]:
                            for target in protocol["targets"]:
                                frontiers.append({"record_type": "frontier", "question_ids": ["Q4", "Q6", "Q7"],
                                    "parent_id": parent["parent_id"], "continuous_field_id": parent["continuous_field_id"],
                                    "grid_size": n, "track": track, "horizon": horizon, "trial_id": identifier,
                                    "family": family, "seed": seed, "norm": norm, "target": target, "status": "FEASIBLE",
                                    "timing_seconds": .0012345678901234, "selected_schedule": protocol["confirm_schedules"][-1],
                                    "selected_error_upper": .00000012345678901234,
                                    "selection_rule": "reference-informed posthoc fixed schedules; not deployment policy"})
    assert len(candidates) == 10200 and len(frontiers) == 32640
    criteria = [{"record_type": "comparison", "question_ids": ["Q4", "Q6", "Q7"], "trial_id": trial["trial_id"],
                 "family": trial["family"], "seed": trial["seed"], "track": track, "norm": norm, "classical": classical,
                 "status": "INCONCLUSIVE", "gates": {"coverage": False, "cost": False}, "criteria_frozen_before_confirmation": True}
                for trial in trials for track in ("discrete", "continuum") for norm in protocol["norms"]
                for classical in ("etdrk4", "gl3_fused")]
    joint = [{"record_type": "comparison", "row_kind": "joint_criterion", "trial_id": trial["trial_id"],
              "family": trial["family"], "track": track, "status": "INCONCLUSIVE", "checkpoint_selection": "TRAINED_CHECKPOINT"}
             for trial in trials for track in ("discrete", "continuum")]
    slots = [{"record_type": "confirmation", "row_kind": "planned_slot", "family": trial["family"],
              "trial_id": trial["trial_id"], "checkpoint_selection": "TRAINED_CHECKPOINT", "status": "COMPLETED"} for trial in trials]
    rows = candidates + frontiers + criteria + joint + slots
    source, report = fixture(tmp_path, rows=rows)
    # Production uses pretty finite JSON; test its full byte cost, not a sparse
    # compact fixture that misses the canonical document's size.
    write_json(source / "rows.json", {"schema": reporting.SCHEMA, "stage": "confirm", "rows": rows})
    assert (source / "rows.json").stat().st_size > 32 << 20
    seal(source)
    result = publish_outputs(report, [source])
    assert result["agenda"]["reporting_complete"], json.loads((report / "outputs/agenda-artifacts.json").read_text())
    assert result["agenda"]["table_rows"]["agenda_confirmation"] == len(candidates) + len(frontiers) + len(slots)
    assert result["agenda"]["table_rows"]["agenda_comparisons"] == len(criteria) + len(joint)
    finish_report(report, state="COMPLETED", exit_code=0, results={"fixture_publication_only": True})
    env = {**os.environ, "PYTHONPATH": str(native), "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, str(ROOT / "scripts/tower_native_validate.py"),
               str(ROOT / ".tower/contracts/outputs.v1.json"), str(report), "--max-bytes", str(64 << 20), "--suite", "agenda"]
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    checked = json.loads(completed.stdout)
    assert checked["valid"] and "agenda" in checked["project_validation_scope"]
    assert all(entry.get("status") not in {"failed", "incomplete"} for entry in checked["outputs"])
