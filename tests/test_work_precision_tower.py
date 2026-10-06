"""Compact CPU work-precision projections retain canonical evidence and budgets."""
import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tdn import tower_analytics as analytics
from tdn.reporting import begin_report


TABLES = ("work_precision", "tolerance_frontiers", "work_precision_checks")


def document():
    candidate = {
        "candidate_id": "fresh-0/gl3/n8", "case_id": "fresh-0", "case_role": "fresh", "variant": "gl3",
        "nsteps": 8, "h": .01, "final_time": .08, "grid": [16], "dtype": "float64", "status": "VALID",
        "trajectory_completed": True, "completed_steps": 8, "finite": True, "in_bounds": True,
        "error_rms": .0001, "error_max": .0002, "error_mean": -.00001, "error_upper_rms": .000101,
        "error_upper_max_estimate": .000202, "reference_accepted": True, "reference_uncertainty_rms": .000001,
        "reference_uncertainty_max_estimate": .000002, "preparation_seconds": .0005,
        "prepared_median_seconds": .003, "prepared_min_seconds": .002, "prepared_max_seconds": .004,
        "setup_inclusive_median_seconds": .0035,
        "timing_repeats": [{"prepared_seconds": t, "setup_inclusive_seconds": t + .0005,
                            "status": "VALID", "work": {"fft_calls": 64, "rhs_evaluations": 0,
                            "reaction_calls": 24, "diffusion_calls": 32}}
                           for t in (.002, .003, .004)],
        "warmup": {"prepared_seconds": .006, "status": "VALID"},
        "work_per_rollout": {"fft_calls": 64, "rhs_evaluations": 0},
        "cache_metadata": {"entries": 6, "cache_hits": 16}, "failure_reason": None,
    }
    failed = copy.deepcopy(candidate)
    failed.update(candidate_id="fresh-0/gl3/n4", nsteps=4, h=.02, status="INVALID", trajectory_completed=False,
                  completed_steps=1, finite=False, in_bounds=False, error_rms=None, error_max=None,
                  error_mean=None, error_upper_rms=None, error_upper_max_estimate=None,
                  prepared_median_seconds=None, failure_reason="nonfinite intermediate")
    failed.pop("setup_inclusive_median_seconds")
    frontier = {
        "frontier_id": "fresh-0/gl3/rms/t1e-03", "case_id": "fresh-0", "case_role": "fresh", "variant": "gl3",
        "norm": "rms", "tolerance": .001, "status": "FEASIBLE", "selected_candidate_id": candidate["candidate_id"],
        "selected_nsteps": 8, "selected_error": .0001, "reference_uncertainty": .000001,
        "adjusted_error": .000101, "prepared_seconds": .003, "setup_inclusive_seconds": .0035,
        "setup_selected_candidate_id": candidate["candidate_id"], "setup_selected_nsteps": 8,
        "setup_selected_seconds": .0035, "setup_selected_error": .0001,
        "setup_selected_adjusted_error": .000101,
        "feasible_candidate_count": 1, "selection_scope": "posthoc_declared_grid",
    }
    infeasible = {**frontier, "frontier_id": "fresh-0/gl3/rms/t1e-06", "tolerance": .000001,
                  "status": "NO_FEASIBLE_CANDIDATE", "selected_candidate_id": None, "selected_nsteps": None,
                  "selected_error": None, "adjusted_error": None, "prepared_seconds": None,
                  "setup_inclusive_seconds": None, "feasible_candidate_count": 0,
                  "setup_selected_candidate_id": None, "setup_selected_nsteps": None,
                  "setup_selected_seconds": None, "setup_selected_error": None,
                  "setup_selected_adjusted_error": None}
    return {
        "schema": "tdn.work-precision/v1", "benchmark_suite": "work-precision", "status": "COMPLETED",
        "computational_status": "COMPLETED", "device": "cpu", "training_attempted": False,
        "training_performed": False, "candidate_rows": [candidate, failed], "frontier_rows": [frontier, infeasible],
        "reference_rows": [{"reference_id": "fresh-0/reference", "case_id": "fresh-0", "case_role": "fresh",
            "reference_accepted": True, "uncertainty_rms": .000001, "uncertainty_max_estimate": .000002,
            "reference_n": 64, "reference_2n": 128, "reference_4n": 256, "difference_n_2n": .00004,
            "difference_2n_4n": .00001, "observed_order": 2., "reference_reason": "converged",
            "attempts": 1, "rhs_evaluations": 1024, "reference_seconds": .04}],
        "parity_rows": [{"parity_id": "fresh-0/identity/float64/gl3", "case_id": "fresh-0", "check": "identity",
            "dtype": "float64", "nodes": 3, "nsteps": 4, "max_difference": 1e-15, "rms_difference": 5e-16,
            "allowed_absolute_difference": 1e-12, "passed": True, "status": "PASS", "failure_reason": None}],
    }


def make_report(tmp_path, payload=None, *, name="work-precision.json"):
    source = tmp_path / "experiment"
    source.mkdir()
    canonical = source / name
    canonical.write_text(json.dumps(document() if payload is None else payload, allow_nan=False))
    report = Path(begin_report(source, name="tdn/work-precision", script="scripts/work_precision.py",
                              report_parent=tmp_path / "reports"))
    return source, report, canonical


def read_rows(report, table):
    with (report / "outputs" / f"{table}.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def resolve(document, pointer):
    for part in pointer.strip("/").split("/"):
        document = document[int(part)] if isinstance(document, list) else document[part.replace("~1", "/").replace("~0", "~")]
    return document


def test_compact_rows_exact_pointers_nulls_logs_contract_and_raw_retention(tmp_path):
    payload = document()
    source, report, canonical = make_report(tmp_path, payload)
    # Summary and panel copies are inventoried without duplicating canonical rows.
    (source / "summary.json").write_bytes(canonical.read_bytes())
    (source / "candidates.json").write_bytes(canonical.read_bytes())
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}

    result = analytics.publish_outputs(report, [source])

    candidates = read_rows(report, "work_precision")
    frontiers = read_rows(report, "tolerance_frontiers")
    checks = read_rows(report, "work_precision_checks")
    assert len(candidates) == len(frontiers) == len(checks) == 2
    assert result["table_rows"]["mechanisms"] == result["table_rows"]["training"] == 0
    assert result["reporting_omission_count"] == 0
    assert result["scientific_outcomes"] == []
    assert candidates[1]["error_rms"] == candidates[1]["prepared_median_seconds"] == "null"
    assert candidates[1]["setup_inclusive_median_seconds"] == ""
    assert candidates[1]["status"] == "INVALID" and candidates[1]["finite"] == "false"
    assert frontiers[1]["status"] == "NO_FEASIBLE_CANDIDATE"
    assert frontiers[1]["selected_candidate_id"] == "null" and frontiers[1]["selected_candidate_record"] == ""
    assert frontiers[0]["selected_candidate_record"] == "/candidate_rows/0"
    assert frontiers[0]["setup_selected_candidate_record"] == "/candidate_rows/0"
    for table in TABLES:
        for row in read_rows(report, table):
            assert row["device"] == "cpu" and row["training_attempted"] == "false"
            original = json.loads((report / row["source_path"]).read_text())
            assert resolve(original, row["source_record"])["case_id"] == row["case_id"]
    for index, row in enumerate(candidates):
        for key in ("timing_repeats", "warmup", "work_per_rollout", "cache_metadata"):
            pointer = row[key + "_record"]
            assert pointer == f"/candidate_rows/{index}/{key}"
            assert resolve(payload, pointer) == payload["candidate_rows"][index][key]
    assert {row["record_type"] for row in checks} == {"reference", "parity"}
    assert checks[0]["reference_accepted"] == "true" and checks[1]["passed"] == "true"
    assert hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in hashes}
    summary, = result["reports"]
    assert summary["unsupported_row_count"] == 0
    assert summary["candidate_status_counts"] == {"VALID": 1, "INVALID": 1, "INCOMPLETE": 0, "FAILED": 0}
    assert summary["scientific_gate_authorized"] is summary["neural_superiority_claim"] is False
    metadata = json.loads((report / "outputs/tables.json").read_text())
    assert "not a deployable step controller" in metadata["work_precision_interpretation"]
    logs = json.loads((report / "logs.json").read_text())["logs"]
    for table in TABLES:
        assert sum(row["id"] == "analytics." + table for row in logs) == 1
    # Reprojection neither doubles registrations nor mutates original evidence.
    analytics.publish_outputs(report, [source])
    logs = json.loads((report / "logs.json").read_text())["logs"]
    assert sum(row["id"].startswith("analytics.") for row in logs) == 3
    contract = json.loads((Path(__file__).resolve().parents[1] / ".tower/contracts/outputs.v1.json").read_text())
    for table in TABLES:
        entry, = [entry for entry in contract["outputs"] if entry["path"] == f"outputs/{table}.csv"]
        assert entry["columns"] == metadata["columns"][table] == analytics.COLUMNS[table]
        assert entry["required"] is False and entry["format"] == "csv"
        assert entry["max_rows"] == 10000 and entry["max_bytes"] == 2 << 20


@pytest.mark.parametrize("changes", [
    {"schema": "tdn.work-precision/v2"}, {"benchmark_suite": "interaction-screen"},
    {"device": "cuda"}, {"training_attempted": True}, {"training_performed": True},
    {"status": "RUNNING"}, {"computational_status": "RUNNING"}, {"candidate_rows": {}},
    {"frontier_rows": None}, {"reference_rows": None}, {"parity_rows": "unsupported"},
])
def test_unsupported_schema_is_explicitly_incomplete(tmp_path, changes):
    payload = {**document(), **changes}
    source, report, canonical = make_report(tmp_path, payload)
    before = canonical.read_bytes()
    result = analytics.publish_outputs(report, [source])
    assert all(result["table_rows"][table] == 0 for table in TABLES)
    assert result["scientific_outcomes"] == ["WORK_PRECISION_INCOMPLETE"]
    assert result["reports"][0]["unsupported_document_count"] == 1
    assert result["reporting_omission_count"] == 1
    assert canonical.read_bytes() == before


@pytest.mark.parametrize("field", ["candidate_rows", "frontier_rows", "reference_rows", "parity_rows"])
@pytest.mark.parametrize("different_id", [False, True])
def test_duplicate_logical_or_declared_identities_are_counted(tmp_path, field, different_id):
    payload = document()
    duplicate = copy.deepcopy(payload[field][0])
    if different_id:
        duplicate[field.removesuffix("_rows") + "_id"] += "-alias"
    payload[field].append(duplicate)
    source, report, _ = make_report(tmp_path, payload)
    result = analytics.publish_outputs(report, [source])
    assert result["reports"][0]["unsupported_row_count"] == 1
    assert result["scientific_outcomes"] == ["WORK_PRECISION_INCOMPLETE"]
    assert result["table_rows"]["work_precision"] == 2
    assert result["table_rows"]["tolerance_frontiers"] == 2
    assert result["table_rows"]["work_precision_checks"] == 2
    assert "duplicate canonical work-precision" in json.loads((report / "outputs/artifacts.json").read_text())["omissions"][0]["reason"]


@pytest.mark.parametrize("field,key,value", [
    ("candidate_rows", "error_rms", {"hidden": .1}),
    ("candidate_rows", "prepared_median_seconds", "fast"),
    pytest.param("candidate_rows", "error_rms", 10 ** 400, id="unrepresentable-number"),
    ("candidate_rows", "finite", 1), ("candidate_rows", "nsteps", True),
    ("candidate_rows", "grid", [16, False]), ("candidate_rows", "timing_repeats", {}),
    ("candidate_rows", "warmup", []), ("candidate_rows", "work_per_rollout", []),
    ("candidate_rows", "cache_metadata", None),
    ("candidate_rows", "device", "cuda"), ("candidate_rows", "status", "COMPLETED"),
    ("frontier_rows", "norm", "unspecified"), ("frontier_rows", "selected_candidate_id", "missing"),
    ("frontier_rows", "adjusted_error", .1), ("frontier_rows", "selection_scope", "deployable"),
    ("frontier_rows", "setup_selected_candidate_id", "missing"),
    ("frontier_rows", "setup_selected_adjusted_error", .1),
    ("reference_rows", "reference_accepted", "yes"), ("reference_rows", "uncertainty_rms", []),
    ("parity_rows", "passed", False), ("parity_rows", "nodes", 3.5),
])
def test_invalid_row_fields_cannot_create_affirmative_evidence(tmp_path, field, key, value):
    payload = document()
    payload[field][0][key] = value
    source, report, _ = make_report(tmp_path, payload)
    result = analytics.publish_outputs(report, [source])
    assert result["reports"][0]["unsupported_row_count"] >= 1
    assert result["scientific_outcomes"] == ["WORK_PRECISION_INCOMPLETE"]
    assert result["reports"][0]["scientific_gate_authorized"] is False


def test_incomplete_failed_rejected_reference_and_unavailable_parity_retained(tmp_path):
    payload = document()
    payload.update(status="INCOMPLETE", computational_status="INCOMPLETE")
    payload["candidate_rows"][1]["status"] = "FAILED"
    payload["reference_rows"][0].update(reference_accepted=False, uncertainty_rms=None,
                                         reference_reason="reference budget exhausted")
    payload["parity_rows"][0].update(status="UNAVAILABLE", passed=None, max_difference=None,
                                    rms_difference=None, failure_reason="budget exhausted")
    source, report, _ = make_report(tmp_path, payload)
    result = analytics.publish_outputs(report, [source])
    assert result["scientific_outcomes"] == ["WORK_PRECISION_INCOMPLETE"]
    assert result["reports"][0]["rejected_reference_count"] == 1
    assert result["reports"][0]["parity_status_counts"]["UNAVAILABLE"] == 1
    assert result["reports"][0]["candidate_status_counts"]["FAILED"] == 1
    checks = read_rows(report, "work_precision_checks")
    assert checks[0]["uncertainty_rms"] == "null"
    assert checks[1]["passed"] == "null"


@pytest.mark.parametrize("status", ["FAILED", "INCOMPLETE"])
def test_preparation_failure_before_warmup_retains_actual_candidate_shape(tmp_path, status):
    from tdn.analysis.work_precision.experiment import _new_candidate

    payload = document()
    candidate = _new_candidate({"case_id": "fresh-0", "case_role": "fresh", "final_time": .08,
                                "grid": [16]}, "gl3", 8, payload["reference_rows"][0])
    candidate.update(status=status, preparation_seconds=.0002, failure_reason="preparation interrupted")
    payload.update(status=status, computational_status=status, candidate_rows=[candidate],
                   frontier_rows=[], parity_rows=[])
    source, report, canonical = make_report(tmp_path, payload)
    before = canonical.read_bytes()

    result = analytics.publish_outputs(report, [source])

    row, = read_rows(report, "work_precision")
    assert row["status"] == status
    assert row["trajectory_completed"] == "false" and row["completed_steps"] == "0"
    assert row["preparation_seconds"] == "0.0002"
    assert row["prepared_median_seconds"] == row["error_rms"] == "null"
    assert row["warmup_record"] == "/candidate_rows/0/warmup"
    assert resolve(payload, row["warmup_record"]) is None
    assert resolve(payload, row["timing_repeats_record"]) == []
    assert resolve(payload, row["work_per_rollout_record"]) == {}
    assert result["reports"][0]["unsupported_row_count"] == 0
    assert result["reporting_omission_count"] == 0
    assert result["scientific_outcomes"] == ["WORK_PRECISION_INCOMPLETE"]
    assert canonical.read_bytes() == before


def test_setup_inclusive_winner_can_differ_from_prepared_winner(tmp_path):
    payload = document()
    candidate = copy.deepcopy(payload["candidate_rows"][0])
    candidate.update(candidate_id="fresh-0/gl3/n16", nsteps=16, prepared_median_seconds=.0032,
                     setup_inclusive_median_seconds=.0033)
    payload["candidate_rows"].append(candidate)
    payload["frontier_rows"][0].update(setup_selected_candidate_id=candidate["candidate_id"],
        setup_selected_nsteps=16, setup_selected_seconds=.0033)
    source, report, _ = make_report(tmp_path, payload)
    result = analytics.publish_outputs(report, [source])
    assert result["reporting_omission_count"] == 0
    row = read_rows(report, "tolerance_frontiers")[0]
    assert row["selected_nsteps"] == "8" and row["setup_selected_nsteps"] == "16"
    assert row["selected_candidate_record"] == "/candidate_rows/0"
    assert row["setup_selected_candidate_record"] == "/candidate_rows/2"


@pytest.mark.parametrize("name,limit", [
    ("work-precision.json", analytics.MAX_WORK_PRECISION_FILE_BYTES),
    ("Work-precision.json", analytics.MAX_FILE_BYTES),
    ("work-precision.JSON", analytics.MAX_FILE_BYTES),
    ("summary.json", analytics.MAX_FILE_BYTES),
])
def test_read_override_is_exact_and_bounded(tmp_path, name, limit):
    source, report, canonical = make_report(tmp_path, name=name)
    original = canonical.read_bytes()
    canonical.write_bytes(original + b" " * (limit - len(original)))
    budget = analytics._Budget()
    assert len(budget.read(canonical, canonical.stat())) == limit
    canonical.write_bytes(canonical.read_bytes() + b" ")
    with pytest.raises(ValueError, match="per-file or total read budget"):
        analytics._Budget().read(canonical, canonical.stat())
    result = analytics.publish_outputs(report, [source])
    assert result["table_rows"]["work_precision"] == 0
    if name == "work-precision.json":
        assert result["scientific_outcomes"] == ["WORK_PRECISION_INCOMPLETE"]
    inventory = json.loads((report / "outputs/artifacts.json").read_text())
    assert inventory["artifacts"][0]["hash_status"] == "not_read_budget"
    assert inventory["observed_read_bytes"] == 0
    assert inventory["limits"]["read_bytes"] == 16 << 20
    assert inventory["limits"]["per_file_read_bytes"] == 1 << 20


@pytest.mark.parametrize("invalid", ["NaN", "Infinity", "1e999"])
def test_nonfinite_source_is_not_projected_or_rewritten(tmp_path, invalid):
    source, report, canonical = make_report(tmp_path)
    raw = canonical.read_text().replace('"error_rms": 0.0001', '"error_rms": ' + invalid, 1)
    canonical.write_text(raw)
    result = analytics.publish_outputs(report, [source])
    assert result["table_rows"]["work_precision"] == 0
    assert result["scientific_outcomes"] == ["WORK_PRECISION_INCOMPLETE"]
    assert canonical.read_text() == raw


def test_expanded_canonical_still_obeys_global_read_and_row_budgets(tmp_path, monkeypatch):
    source, report, canonical = make_report(tmp_path)
    size = canonical.stat().st_size
    budget = analytics._Budget()
    budget.read_bytes = analytics.MAX_READ_BYTES - size
    assert budget.read(canonical, canonical.stat()) == canonical.read_bytes()
    with pytest.raises(ValueError, match="per-file or total read budget"):
        budget.read(canonical, canonical.stat())
    monkeypatch.setattr(analytics, "MAX_ROWS", 1)
    result = analytics.publish_outputs(report, [source])
    assert sum(result["table_rows"][table] for table in TABLES) == 1
    assert result["reports"][0]["supported_candidate_count"] == 2
    assert result["reporting_omission_count"] == 1
    assert len(json.loads(canonical.read_text())["candidate_rows"]) == 2


def test_realistic_full_matrix_stays_compact_with_raw_repetitions(tmp_path):
    payload = document()
    candidate, frontier = payload["candidate_rows"][0], payload["frontier_rows"][0]
    payload["candidate_rows"], payload["frontier_rows"] = [], []
    # 24 cases x 7 methods x 6 step counts, plus 2 norms x 4 tolerances.
    for case in range(24):
        for method in range(7):
            case_id, variant = f"fresh-{case}", f"method-{method}"
            for nsteps in (4, 8, 16, 32, 64, 128):
                row = copy.deepcopy(candidate)
                row.update(case_id=case_id, variant=variant, nsteps=nsteps,
                           candidate_id=f"{case_id}/{variant}/n{nsteps}")
                # Independent setup-inclusive and prepared repeats retain counters.
                row["timing_repeats"] *= 2
                payload["candidate_rows"].append(row)
            for norm in ("rms", "max"):
                for tolerance in (.001, .002, .004, .008):
                    row = {**frontier, "case_id": case_id, "variant": variant, "norm": norm,
                           "tolerance": tolerance, "selected_candidate_id": f"{case_id}/{variant}/n8",
                           "setup_selected_candidate_id": f"{case_id}/{variant}/n8",
                           "frontier_id": f"{case_id}/{variant}/{norm}/t{tolerance}"}
                    payload["frontier_rows"].append(row)
    source, report, canonical = make_report(tmp_path, payload)
    raw = canonical.read_bytes()
    assert analytics.MAX_FILE_BYTES < len(raw) <= analytics.MAX_WORK_PRECISION_FILE_BYTES
    with pytest.raises(ValueError, match="JSON value budget exceeded"):
        analytics._decode(raw)
    before = hashlib.sha256(raw).hexdigest()
    result = analytics.publish_outputs(report, [source])
    assert result["table_rows"]["work_precision"] == 1008
    assert result["table_rows"]["tolerance_frontiers"] == 1344
    assert result["reporting_omission_count"] == 0
    assert sum(path.stat().st_size for path in report.rglob("*") if path.is_file()) < 8 << 20
    assert all((report / "outputs" / f"{table}.csv").stat().st_size <= 2 << 20 for table in TABLES)
    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == before
    inventory = json.loads((report / "outputs/artifacts.json").read_text())
    assert inventory["limits"]["json_values"] == 100000
    assert inventory["limits"]["json_values_overrides"] == {"work-precision.json": 300000}
