"""Per-seed Tower projections preserve declared endpoints and bounded source I/O."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tdn import tower_analytics as analytics
from tdn.reporting import begin_report
from tdn.research.neural_comparison import summarize_neural_comparisons
from tdn.research.replication_protocol import DEFAULT, FAMILIES, make_protocol, validate_config
from tdn.research.replication_summary import summarize_replication


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def read_rows(report, name):
    with (report / "outputs" / f"{name}.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def fixture(tmp_path, *, gpu=False):
    source = tmp_path / "science"
    source.mkdir()
    report = Path(begin_report(source, name="tdn/neural-replication", script="scripts/research.py",
                               report_parent=tmp_path / "reports"))
    protocol = make_protocol(validate_config(DEFAULT, smoke=True), smoke=True, software={}, command=[])
    records, blocks = [], []
    for plan in protocol["replicates"]:
        training = []
        for family in FAMILIES:
            failed = plan["training_seed"] == 74031 and family == "fno_split"
            initialized = plan["training_seed"] == 74021 and family == "unet_split"
            failure = {"parent_id": "validation-parent", "h": .08, "step": 2,
                       "error": "Invalid validation intermediate"}
            row = {**plan, "family": family, "status": "NUMERICAL_FAILURE" if failed else "COMPLETED",
                   "steps": 2, "selected_step": None if failed else 0 if initialized else 1,
                   "selected_parameters_changed": not (failed or initialized), "parameter_count": 242,
                   "training_seconds": 1., "history": [{"step": 2, "validation_loss": None if failed else 1.,
                        "admissible_validation": not failed, "failures": [failure] if failed else []}],
                   "validation_failures": [failure] if failed else [],
                   "failure_context": "checkpoint_selection" if failed else None,
                   "error": "No admissible checkpoint" if failed else None}
            training.append(row)
            records.append(row)
            if not gpu:
                put(source / "replicates" / plan["replicate_id"] / "training" / f"{family}.json", row)
        for block_index in range(3):
            frontier, heldout = [], []
            parents = [row for row in protocol["parents"] if row.get("diagnostic_block") == block_index]
            for parent_index, parent in enumerate(parents):
                for family in FAMILIES:
                    failed = plan["training_seed"] == 74031 and family == "fno_split"
                    for h in protocol["config"]["benchmark_steps"]:
                        if family == "generic_mlp" and block_index == parent_index == 0 and h == .32:
                            continue  # Missing is distinct from invalid and from a tolerance pass.
                        frontier.append({"parent_id": parent["parent_id"], "family": family, "h": h,
                            "device": "cuda" if gpu else "cpu", "status": "NUMERICAL_FAILURE" if failed else "COMPLETED",
                            "error": None if failed else .001, "error_upper": None if failed else .00101,
                            "feasible": not failed, "timing": None if failed else {
                                "wall_seconds_median": .0012 if family == "reaction_clock" else .002}})
                    for h in protocol["config"]["heldout_horizons"]:
                        invalid = failed or (family == "reaction_clock" and block_index == parent_index == 0 and h == .03)
                        upper = .004 if family == "reaction_clock" and block_index == parent_index == 0 and h == .11 else .00101
                        heldout.append({"parent_id": parent["parent_id"], "family": family, "h": h,
                            "device": "cuda" if gpu else "cpu", "status": "NUMERICAL_FAILURE" if invalid else "COMPLETED",
                            "one_step": {"rms": None if invalid else upper - .00001,
                                         "upper": None if invalid else upper},
                            "two_step": {"rms": None if invalid else .001,
                                         "upper": None if invalid else .00101}})
            context = {**plan, "diagnostic_block": block_index}
            neural = summarize_neural_comparisons(frontier, heldout, list(FAMILIES), training)
            blocks.append({**context, "frontier": frontier, "heldout": heldout, "neural": neural})
            directory = source / "replicates" / plan["replicate_id"] / "blocks" / f"block-{block_index}"
            put(directory / "frontier.json", {"replication": context, "rows": frontier})
            put(directory / "heldout.json", {"replication": context, "rows": heldout})
            put(directory / "neural-comparisons.json", {"replication": context, **neural})
    summary = summarize_replication(protocol, records, blocks)
    summary.update(status="COMPLETED", device="cuda" if gpu else "cpu", training_attempted=not gpu,
                   actual_neural_training=not gpu)
    put(source / "replication.json", summary)
    # The readable root summary embeds endpoints; this must not double the tables.
    put(source / "summary.json", {**summary, "source_training" if gpu else "training": records})
    return source, report, summary


def test_seed_block_context_and_transient_endpoints_survive_projection(tmp_path):
    source, report, summary = fixture(tmp_path)
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob("*.json")}
    result = analytics.publish_outputs(report, [source])
    assert result["reporting_omission_count"] == 0
    assert len(read_rows(report, "replication")) == 15
    assert len(read_rows(report, "replication_comparisons")) == 12
    assert len(read_rows(report, "training")) == 30  # embedded summary does not duplicate histories
    for table in ("frontier", "heldout", "neural_comparisons", "neural_accuracy"):
        records = read_rows(report, table)
        assert {row["replicate_id"] for row in records} == {"seed-74011", "seed-74021", "seed-74031"}
        assert {row["diagnostic_block"] for row in records} == {"0", "1", "2"}
        for row in records:
            assert row["replicate_id"] == "seed-" + row["training_seed"]
            assert int(row["sample_schedule_seed"]) == int(row["training_seed"]) + 1
            context = json.loads((report / row["source_path"]).read_text())["replication"]
            assert int(row["diagnostic_block"]) == context["diagnostic_block"]
    endpoints = read_rows(report, "replication")
    clock = next(row for row in endpoints if row["family"] == "reaction_clock" and row["training_seed"] == "74011")
    assert clock["parent_count"] == "9"
    assert clock["rollout_expected"] == clock["rollout_pass"] == "36"
    assert clock["heldout_one_expected"] == "18"
    assert clock["heldout_one_invalid"] == "1" and clock["heldout_one_pass"] == "16"
    assert clock["heldout_two_invalid"] == "1" and clock["heldout_two_pass"] == "17"
    assert clock["feasible_parent_count"] == "9" and clock["robust_joint_parent_pass_count"] == "8"
    assert json.loads(clock["worst_heldout_one"])["h"] == .11
    generic = next(row for row in endpoints if row["family"] == "generic_mlp")
    assert generic["rollout_missing"] == "1" and generic["rollout_pass"] == "35"
    failed = next(row for row in endpoints if row["family"] == "fno_split" and row["training_seed"] == "74031")
    assert failed["selection"] == "TRAINING_FAILURE" and failed["rollout_invalid"] == "36"
    initialized = next(row for row in endpoints if row["family"] == "unet_split" and row["training_seed"] == "74021")
    assert initialized["selection"] == "SELECTED_INITIALIZATION" and initialized["trained_checkpoint"] == "false"
    comparisons = read_rows(report, "replication_comparisons")
    init_pair = next(row for row in comparisons if row["baseline_family"] == "unet_split" and row["training_seed"] == "74021")
    assert init_pair["eligible"] == "9" and init_pair["trained_pair"] == "false"
    assert init_pair["heldout_one_expected"] == "18" and init_pair["heldout_one_ineligible"] == "1"
    science = [row for row in result["reports"] if row["kind"] == "neural_replication"]
    assert len(science) == 1 and science[0]["diagnostic_parent_count"] == summary["diagnostic_parent_count"] == 9
    assert science[0]["scientific_gate_authorized"] is False
    assert "not independent" in result["neural_observation_scope"]
    training = next(row for row in read_rows(report, "training")
                    if row["family"] == "fno_split" and row["training_seed"] == "74031" and row["record_type"] == "final")
    assert training["failure_context"] == "checkpoint_selection"
    assert json.loads(training["validation_failures"])[0]["parent_id"] == "validation-parent"
    assert hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in hashes}


def test_gpu_source_checkpoints_do_not_imply_retraining(tmp_path):
    source, report, _ = fixture(tmp_path, gpu=True)
    result = analytics.publish_outputs(report, [source])
    training = read_rows(report, "training")
    assert len(training) == 30
    assert {row["record_type"] for row in training} == {"source_checkpoint", "source_history"}
    assert len(read_rows(report, "replication")) == 15
    science = next(row for row in result["reports"] if row["kind"] == "neural_replication")
    assert science["training_attempted"] is False and science["actual_neural_training"] is False


def test_contract_has_optional_exact_replication_tables_and_context_columns():
    contract = json.loads((Path(__file__).parents[1] / ".tower/contracts/outputs.v1.json").read_text())
    entries = {row["path"]: row for row in contract["outputs"]}
    for table in analytics.COLUMNS:
        row = entries[f"outputs/{table}.csv"]
        assert row["required"] is False and row["format"] == "csv"
        assert row["columns"] == analytics.COLUMNS[table]
        expected_bytes = (analytics.MAX_COMPACT_SPATIAL_TABLE_BYTES if table in analytics.COMPACT_SPATIAL_TABLES
                          else analytics.MAX_MECHANISM_TABLE_BYTES if table == "mechanisms" else analytics.MAX_TABLE_BYTES)
        expected_rows = (analytics.MAX_COMPACT_SPATIAL_ROWS if table in analytics.COMPACT_SPATIAL_TABLES
                         else analytics.MAX_MECHANISM_ROWS if table == "mechanisms" else analytics.MAX_ROWS)
        assert row["max_bytes"] == expected_bytes and row["max_rows"] == expected_rows
    for table in ("training", "frontier", "heldout", "neural_comparisons", "neural_accuracy"):
        assert set(analytics.REPLICATION_CONTEXT) <= set(analytics.COLUMNS[table])


def test_invalid_seed_context_is_omitted_instead_of_conflating_records(tmp_path):
    source, report, summary = fixture(tmp_path)
    summary["endpoint_rows"][0]["replicate_id"] = []
    summary["endpoint_rows"][1].update(source_path="../forged.json", source_record="/forged")
    summary["comparison_rows"][0]["training_seed"] = 74021
    put(source / "replication.json", summary)
    result = analytics.publish_outputs(report, [source])
    assert len(read_rows(report, "replication")) == 14
    assert len(read_rows(report, "replication_comparisons")) == 11
    assert result["reporting_omission_count"] == 2
    valid = read_rows(report, "replication")[0]
    assert valid["source_path"].endswith("/replication.json")
    assert valid["source_record"] == "/endpoint_rows/1"


def test_replication_projection_retains_global_row_budget(tmp_path, monkeypatch):
    _, _, summary = fixture(tmp_path)
    budget = analytics._Budget()
    projection = analytics._Projection(budget, set())
    monkeypatch.setattr(analytics, "MAX_ROWS", 8)
    projection.consume(Path("replication.json"), "../replication.json", summary)
    assert sum(len(rows) for rows in projection.tables.values()) == 8
    assert budget.omission_count == 1


def test_csv_exact_utf8_byte_limit_remains_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(analytics, "MAX_TABLE_BYTES", 12)
    budget = analytics._Budget()
    target = tmp_path / "unicode.csv"
    count = analytics._atomic_csv(target, [{"x": "😀"}] * 3, ["x"], budget)
    assert count == 2 and target.stat().st_size == 12
    assert budget.omission_count == 1


def test_complete_replication_frontier_fits_actual_two_mib_csv_budget(tmp_path):
    # 3 seeds x 3 blocks x 9 parents x 10 methods x 4 steps. Existing source
    # shards obey their per-file JSON limits while the unified CSV retains them.
    rows = [{"source_path": "../" + "r" * 240 + "/frontier.json", "source_record": f"/rows/{index % 360}",
             "replicate_id": f"seed-{74011 + 10 * (index // 1080)}", "training_seed": 74011 + 10 * (index // 1080),
             "sample_schedule_seed": 74012 + 10 * (index // 1080), "diagnostic_block": (index // 360) % 3,
             "parent_id": "diagnostic-stiff-mixed_frequency-34207", "family": "reaction_clock",
             "h": .32, "status": "COMPLETED", "device": "cuda", "error": .001,
             "error_upper": .0010001, "feasible": True, "wall_seconds_median": .00314,
             "cuda_event_seconds_median": .00312, "peak_allocated_bytes": 1804800,
             "peak_reserved_bytes": 2097152, "host_process_peak_rss_bytes": 965042176,
             "timing_scope": "full operation including features, solver and inference"} for index in range(3240)]
    budget = analytics._Budget()
    target = tmp_path / "frontier.csv"
    assert analytics._atomic_csv(target, rows, analytics.COLUMNS["frontier"], budget) == 3240
    assert analytics.MAX_TABLE_BYTES // 4 < target.stat().st_size <= analytics.MAX_TABLE_BYTES
    assert budget.omission_count == 0


@pytest.mark.parametrize("version,protocol_version", [(2, 2), (1, 1)])
def test_unsupported_replication_versions_are_explicit(tmp_path, version, protocol_version):
    source, report, summary = fixture(tmp_path)
    summary.update(version=version, protocol_version=protocol_version)
    put(source / "replication.json", summary)
    result = analytics.publish_outputs(report, [source])
    assert read_rows(report, "replication") == read_rows(report, "replication_comparisons") == []
    assert result["reporting_omission_count"] == 1
