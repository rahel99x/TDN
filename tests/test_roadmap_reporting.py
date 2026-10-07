"""Read-only Tower pages retain every experiment, check and source pointer."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tdn import roadmap_reporting as reporting
from tdn.analysis.roadmap.core import Context, check, write_reviews
from tdn.analysis.roadmap.protocol import build_protocol
from tdn.reporting import begin_report
from tdn.research.protocol import digest
from tdn.runtime.metadata import write_json
from tdn.tower_analytics import publish_outputs

ROOT = Path(__file__).resolve().parents[1]


def test_csv_fast_path_preserves_controls_unicode_formula_safety_and_json_values():
    from tdn.premix_reporting import _cell as existing_cell
    cases = ["", "ordinary text", "123.5", "-123.5", " =SUM(A1:A2)", "+1", "@SUM(A1)",
             "\u2003\u00a0=SUM(A1)", "\u2028\u2029", "naïve λ 漢字", None, True, False,
             7, -1.5, {"a/b~c": [1, None]}, ["line\nnext", 2]]
    cases.extend("prefix" + chr(code) + "suffix" for code in range(32))
    assert all(reporting._cell(value) == existing_cell(value) for value in cases)
    assert reporting._cell("\u2003=SUM(A1)") == "'\u2003=SUM(A1)"
    assert reporting._cell("a\x00\n\r\tb") == "a\\u0000\\n\\r\\tb"
    with pytest.raises(ValueError):
        reporting._cell(float("nan"))


def fixture(tmp_path, count=3, *, completed=True):
    source = tmp_path / "audit"
    source.mkdir()
    protocol = build_protocol("smoke")
    ctx = Context(protocol, "audit", source, {}, "cpu", SimpleNamespace(check=lambda: None))
    for index in range(count):
        ctx.record(f"case/{index}", ["M02", "M23"], combination_ids=["C3"],
            metrics={"parameters": 12, "median_seconds": .001 + index * .00001,
                     "extra": {"a/b~c": [1., 2.]}, "cost": {"fft": 9}},
            config={"family": "source", "seed": 7, "grid": 8, "dtype": "float64"},
            checks=[check("numerical-order", 3.1, 2.9, "ge", category="math"),
                    check("gap", index % 2, 0, "eq", category="gap"),
                    check("unavailable-population-risk", None, .01, category="utility")])
    write_reviews(source, ctx.rows)
    write_json(source / "protocol.json", protocol)
    write_json(source / "summary.json", {"schema": protocol["schema"], "stage": "audit",
        "status": "COMPLETED" if completed else "FAILED", "experiment_count": count, "scientific_outcome": "NA"})
    if completed:
        seal(source, protocol)
    report = Path(begin_report(source, name="TDN/roadmap/audit", script="scripts/roadmap.py",
        parameters={"benchmark_suite": "roadmap"}, report_parent=tmp_path / "tower"))
    return source, report, protocol


def seal(source, protocol):
    manifest = {"schema": "tdn.roadmap-science-manifest/v1", "stage": "audit", "protocol_sha256": digest(protocol),
        "source_tree_sha256": "a" * 64,
        "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()
                      if p.is_file() and p.name not in ("science_manifest.json", "COMPLETED", "stage.json", "workflow-seal.json")}}
    write_json(source / "science_manifest.json", manifest)
    (source / "COMPLETED").write_text(digest(manifest) + "\n")


def index_and_pages(report, kind=None):
    index = json.loads((report / "outputs/roadmap-tables.json").read_text())
    pages = []
    for catalog in index["page_catalogs"]:
        path = report / catalog["path"]
        assert catalog["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        pages.extend(json.loads(path.read_text())["pages"])
    return index, [p for p in pages if kind is None or p["table"] == kind]


def table(report, kind):
    _, pages = index_and_pages(report, kind)
    result = []
    for page in pages:
        with (report / page["path"]).open(newline="") as stream:
            result.extend(csv.DictReader(stream))
    return result


def resolve(report, row, key="source_record"):
    path = report / row["source_path"]
    if row.get("source_line"):
        value = json.loads(path.read_text().splitlines()[int(row["source_line"]) - 1])
    else:
        value = json.loads(path.read_text())
    for part in row[key].split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def test_all_experiments_checks_values_and_metric_points_survive_pagination(tmp_path):
    source, report, _ = fixture(tmp_path, count=301)
    originals = {p.name: p.read_bytes() for p in source.iterdir()}
    result = publish_outputs(report, [source])
    assert result["roadmap"]["reporting_complete"], result
    assert "agenda" not in result and not (report / "outputs/agenda-tables.json").exists()
    assert result["roadmap"]["table_rows"]["experiments"] == 301
    assert result["roadmap"]["table_rows"]["checks"] == 903
    index, pages = index_and_pages(report)
    assert index["expected_rows"] == index["published_rows"]
    assert all(p["rows"] <= 128 and p["bytes"] <= 262144 for p in pages)
    assert all((report / p["path"]).stat().st_size == p["bytes"] for p in pages)
    rows = table(report, "experiments")
    assert len(rows) == 301 and all(r["evidence_status"] == "VERIFIED_CANONICAL" for r in rows)
    assert resolve(report, rows[123])["experiment_id"] == "case/123"
    checks = table(report, "checks")
    assert all(resolve(report, row)["check_id"] == row["check_id"] for row in checks)
    assert all(row["proof_status"] == "NOT_A_PROOF" for row in checks)
    values = table(report, "values")
    nested = next(r for r in values if r["field"] == "/metrics/extra/a~1b~0c/1")
    assert resolve(report, nested) == 2.
    points = []
    for p in pages:
        if p["table"] == "metrics":
            points += [json.loads(line) for line in (report / p["path"]).read_text().splitlines()]
    assert len(points) == 301 and all(len(p["metrics"]) <= 64 for p in points)
    assert all(resolve(report, p)["row_sha256"] == p["row_sha256"] for p in points)
    assert originals == {p.name: p.read_bytes() for p in source.iterdir()}
    logs = json.loads((report / "logs.json").read_text())["logs"]
    assert len(logs) < 32
    inventory = json.loads((report / "outputs/artifacts.json").read_text())["artifacts"]
    canonical = next(r for r in inventory if r["path"].endswith("/rows.json"))
    assert canonical["hash_status"] == "computed_roadmap_projection"


@pytest.mark.parametrize("mutation", ["rows", "count", "marker", "score", "duplicate", "unsealed", "wrapper", "nonfinite", "stream"])
def test_invalid_or_partial_evidence_is_visible_and_never_credited(tmp_path, mutation):
    source, report, protocol = fixture(tmp_path)
    if mutation in ("rows", "score", "duplicate", "nonfinite"):
        rows = json.loads((source / "rows.json").read_text())
        if mutation == "rows":
            rows["rows"][0]["metrics"]["parameters"] = 999
        elif mutation == "score":
            rows["rows"][0]["assessment"]["score_1_100"] = 100
        elif mutation == "duplicate":
            rows["rows"] += rows["rows"]
        else:
            (source / "rows.json").write_text('{"schema":"tdn.roadmap-rows/v1","rows":[NaN]}')
        if mutation != "nonfinite":
            write_json(source / "rows.json", rows)
    elif mutation == "count":
        summary = json.loads((source / "summary.json").read_text())
        summary["experiment_count"] = 999
        write_json(source / "summary.json", summary)
        seal(source, protocol)
    elif mutation == "marker":
        (source / "COMPLETED").write_text("0" * 64)
    elif mutation == "unsealed":
        (source / "COMPLETED").unlink()
    elif mutation == "stream":
        (source / "rows.jsonl").write_text("")
    else:
        write_json(source / "stage.json", {"status": "FAILED"})
    result = reporting.publish_roadmap_outputs(report, [source])
    assert not result["reporting_complete"] and result["reporting_omission_count"] > 0
    assert all(row["verdict"] == "NA" and row["score_1_100"] == "1" for row in table(report, "experiments"))


def test_interrupted_stream_only_rows_have_exact_lineage_without_success_claim(tmp_path):
    source, report, _ = fixture(tmp_path, completed=False)
    (source / "rows.json").unlink()
    result = reporting.publish_roadmap_outputs(report, [source])
    assert result["table_rows"]["experiments"] == 3 and not result["reporting_complete"]
    rows = table(report, "experiments")
    assert all(r["source_path"].endswith("rows.jsonl") and r["source_line"] for r in rows)
    assert resolve(report, rows[2])["experiment_id"] == "case/2"
    assert all(r["verdict"] == "NA" for r in rows)


def test_checkpoints_remain_unloaded_declared_artifacts(tmp_path):
    source, report, protocol = fixture(tmp_path)
    path = source / "checkpoint.pt"
    path.write_bytes(b"never unpickle this" * 100000)
    seal(source, protocol)
    result = reporting.publish_roadmap_outputs(report, [source])
    assert result["reporting_complete"]
    row = next(r for r in table(report, "artifacts") if r["source_path"].endswith("checkpoint.pt"))
    assert row["hash_status"] == "declared_unverified" and row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_wrapper_seal_and_mechanism_summary_have_independent_source_lineage(tmp_path):
    source, report, protocol = fixture(tmp_path)
    record = {"id": "M02", "kind": "mechanism", "name": "finite-h defect", "verdict": "BAD", "score_1_100": 26,
        "evidence_coverage": .7, "experiment_count": 3, "gap_assessment": {"verdict": "BAD"},
        "math_assessment": {"verdict": "GOOD"}, "failed_checks": ["large-step"], "na_checks": ["population"],
        "gaps": ["D05"], "mathematical_significance": "Finite numerical evidence; not a theorem"}
    write_json(source / "mechanism_summary.json", {"records": [record]})
    write_json(source / "execution.json", {"software": {"source_tree_sha256": "a" * 64}})
    write_json(source / "stage.json", {"status": "COMPLETED"})
    seal(source, protocol)
    write_json(source / "workflow-seal.json", {"schema_version": 1, "protocol_sha256": digest(protocol),
        "files": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                  for name in ("execution.json", "stage.json", "protocol.json", "science_manifest.json")}})
    result = reporting.publish_roadmap_outputs(report, [source])
    assert result["reporting_complete"]
    row = table(report, "mechanisms")[0]
    assert resolve(report, row) == record and resolve(report, row, "failed_checks_record") == ["large-step"]
    assert row["verdict"] == "BAD" and row["score_1_100"] == "26"
    record["verdict"] = "GOOD"
    write_json(source / "mechanism_summary.json", {"records": [record]})
    result = reporting.publish_roadmap_outputs(report, [source])
    assert not result["reporting_complete"] and table(report, "mechanisms")[0]["verdict"] == "NA"


def test_page_ceiling_is_an_explicit_omission_with_exact_partial_counts(tmp_path, monkeypatch):
    source, report, _ = fixture(tmp_path, count=130)
    monkeypatch.setattr(reporting, "MAX_PAGES", 1)
    result = reporting.publish_roadmap_outputs(report, [source])
    index, pages = index_and_pages(report)
    assert not result["reporting_complete"] and index["omission_count"] > 0
    assert index["published_rows"]["experiments"] == 128 and index["expected_rows"]["experiments"] == 130
    assert len(pages) == 1


def test_republication_is_idempotent_in_logs_and_preserves_failed_rows(tmp_path):
    source, report, _ = fixture(tmp_path)
    first = reporting.publish_roadmap_outputs(report, [source])
    logs = (report / "logs.json").read_bytes()
    second = reporting.publish_roadmap_outputs(report, [source])
    assert first["table_rows"] == second["table_rows"] and first["verdicts"] == second["verdicts"]
    assert logs == (report / "logs.json").read_bytes()


def test_existing_native_validation_command_selects_complete_roadmap_contracts(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("roadmap_tower_tools", ROOT / "scripts/tower.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source, report, _ = fixture(tmp_path)
    import sys
    monkeypatch.setattr(module, "tower_profile", lambda root, report: ("fedora", root / ".tower/fedora-slurm.json"))
    monkeypatch.setattr(module.shutil, "which", lambda _: sys.executable)
    monkeypatch.setattr(module, "tower_interpreter", lambda _: (sys.executable, None))
    command, options = module.native_validation_command(ROOT, report)
    assert command[-4:] == ["--max-bytes", str(64 << 20), "--suite", "roadmap"]
    assert options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_full_campaign_serialization_shapes_fit_explicit_reader_and_page_reserves():
    """A protocol expansion must retain room for all scientific row families.

    Full-size publication is also exercised by the private 70k-row integration
    audit. This ordinary allocation test checks its production shapes without
    recreating gigabytes of redundant test artifacts on every GPU workflow.
    """
    protocol = build_protocol("full")
    learned = len(protocol["models"]) * len(protocol["seeds"])
    parents = [p for p in protocol["parents"] if p["split"] == "confirmation"]
    cells = (len(parents) + sum(p["parent_id"] in protocol["continuum_parent_ids"] for p in parents)) * len(protocol["grids"])
    methods = learned + 5
    counts = {"confirm": cells * len(protocol["confirm_schedules"]) * methods,
        "all-schedules": cells * methods, "matched": cells * len(protocol["targets"]) * learned,
        "learned-structure": methods,
        "paired-inference": len(protocol["models"]) * 4 * 2 * len(protocol["targets"])}
    fixtures = json.loads((ROOT / "tests/fixtures/roadmap-reporting-shapes.json").read_text())["rows_by_kind"]
    source_bytes = output_bytes = estimated_pages = 0
    for kind, count in counts.items():
        row = fixtures[kind]
        row["protocol_sha256"] = digest(protocol)
        row["profile"] = "full"
        # Full timings have three samples and schedules can have fifteen steps.
        # This pads every row, including families that need no such arrays.
        row["metrics"]["full_timing_samples_fixture"] = [.001, .002, .003]
        row["effective_config"]["full_schedule_fixture"] = [.03, .07, .17] * 5
        row["row_sha256"] = digest({key: value for key, value in row.items() if key != "row_sha256"})
        source_bytes += len(json.dumps(row, separators=(",", ":"))) * count
        source = {"seal_status": "VERIFIED_CANONICAL", "source_path": "../../confirm/rows.json",
            "source_sha256": "a" * 64, "stage": "confirm", "protocol_sha256": digest(protocol), "reason": "fixture"}
        projected = reporting._projection([(row, "/rows/100000", "")], source, reporting.Budget(), 0.)
        for name, items in zip(("experiments", "checks", "values"), projected[:3]):
            raw = reporting._csv_bytes(reporting.COLUMNS[name], items, header=False)
            output_bytes += len(raw) * count
            estimated_pages += max((len(items) * count + 127) // 128,
                                   (len(raw) * count + reporting.PAGE_BYTES - 1) // reporting.PAGE_BYTES)
        output_bytes += len(json.dumps(projected[3][0])) * count
        estimated_pages += (count + 127) // 128
    assert sum(counts.values()) < 100000
    # Add 20% safety over the largest retained row shape in each family.
    assert source_bytes * 1.2 < reporting.MAX_SOURCE_BYTES
    assert source_bytes * 2.4 < reporting.MAX_READ_BYTES
    assert output_bytes * 1.2 < reporting.MAX_OUTPUT_BYTES
    assert estimated_pages * 1.2 < reporting.MAX_PAGES


def test_native_artifact_reader_can_read_every_exported_page_and_metric_record(tmp_path):
    native = ROOT / ".runtime/tower-current"
    if not (native / "tower/artifact_pages.py").is_file():
        pytest.skip("Optional unchanged Tower checkout is unavailable")
    source, report, _ = fixture(tmp_path, count=140)
    reporting.publish_roadmap_outputs(report, [source])
    # Load unchanged upstream package in a child process to avoid touching the
    # application's package environment or importing scripts/tower.py.
    import os
    import subprocess
    import sys
    program = r'''
import importlib.util, json, pathlib, sys
from tower.artifact_pages import read_page
from tower.artifacts import load_contract, validate_contract
from tower.metrics import _record
root=pathlib.Path(sys.argv[1]); index=json.loads((root/'outputs/roadmap-tables.json').read_text())
for row in index['output_contracts']:
    contract=load_contract(root/row['path'])
    result=validate_contract(contract,root,max_bytes=64<<20,max_entries=4096)
    assert result['valid'], result
    for output in contract['outputs']:
        page=read_page(str(root),output)
        assert page['status']=='ready',page
        assert not page['has_next'] and not page['truncated'],page
for catalog in index['page_catalogs']:
    for page in json.loads((root/catalog['path']).read_text())['pages']:
        if page['table']=='metrics':
            for line in (root/page['path']).read_text().splitlines(): _record(json.loads(line))
spec=importlib.util.spec_from_file_location('native_validator',sys.argv[2])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=module.validate_roadmap_pages(root,load_contract,validate_contract)
assert result['valid'] and result['page_count']==index['page_count'],result
first=json.loads((root/index['page_catalogs'][0]['path']).read_text())['pages'][0]
with (root/first['path']).open('a') as stream: stream.write('tampered')
assert not module.validate_roadmap_pages(root,load_contract,validate_contract)['valid']
print('native pages and metrics verified')
'''
    response = subprocess.run([sys.executable, "-c", program, str(report), str(ROOT / "scripts/tower_native_validate.py")], text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(native), "PYTHONDONTWRITEBYTECODE": "1"}, cwd=native, timeout=60)
    assert response.returncode == 0, response.stdout + response.stderr
