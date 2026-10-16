#!/usr/bin/env python3
"""Read-only native P1R job 2951 audit against immutable git 4d0e049.

Uses only stdlib/git. It hashes retained evidence, checks provenance and
selection/access/metric records, and runs no uploaded code, fit or teacher.
"""
from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

REVISION = "4d0e0498ebb4f218b58934227aebc7619be2d165"
PRELAUNCH = "results/response-preconditioner-regularization-verification-20261015/"
FINGERPRINT = "ff37f88c9dcbdf394a562c0da1ef1765e4f1c1dd8b191940179786de67c66358"
P1 = "results/native-response-preconditioner-scaling-2944/"
P1_ARCHIVE_SHA = "caaacb862df0e272763340b69f523e66884c40e32013cd1ec6f0ff46eb89b6fb"
P1_INDEX_SHA = "43e44377ffb99f0ac4c056132b551e17736b61f35af26352b1650ec1692480b3"
UPLOAD_INDEX_SHA = "243fe99762ebdc8f838116c367df9437bac803901367706a28d2cd690268d8ae"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "evidence", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--upload-index", type=Path)
    args = parser.parse_args()
    root, base = args.repository.resolve(), args.evidence.resolve()
    checks, failures = {}, []

    def require(name, value):
        if not value:
            failures.append(name)
        return bool(value)

    def safe(relative):
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative or path.as_posix() != relative:
            raise ValueError("Unsafe evidence-relative path: "+relative)
        result = base / relative
        if not result.resolve().is_relative_to(base) or any(p.is_symlink() for p in (result, *result.parents)):
            raise ValueError("Evidence path escaped root or traverses symlink")
        return result

    def raw(path):
        return safe(path).read_bytes()

    def read(path):
        return json.loads(raw(path))

    def jsonl(path):
        return [json.loads(line) for line in raw(path).decode().splitlines()]

    def git(*parts):
        return subprocess.check_output(["git", *parts], cwd=root)

    def blob(path):
        return git("show", REVISION+":"+path)

    def blobs_for(paths):
        """One git process for the small source blobs; never imports their code."""
        paths = sorted(set(paths))
        result = subprocess.run(["git", "cat-file", "--batch"], cwd=root,
            input="".join(REVISION+":"+p+"\n" for p in paths).encode(), stdout=subprocess.PIPE, check=True).stdout
        offset, answer = 0, {}
        for path in paths:
            end = result.index(b"\n", offset)
            header = result[offset:end].split()
            if len(header) != 3 or header[1] != b"blob":
                raise ValueError("Expected immutable git blob: "+path)
            size = int(header[2]); offset = end+1
            answer[path] = result[offset:offset+size]
            offset += size+1
        if offset != len(result):
            raise ValueError("Unexpected git batch output")
        return answer

    native, run, contract = read("outputs/native/status.json"), read("run.json"), read("outputs/native/contract.json")
    files = git("ls-tree", "-r", "--name-only", REVISION).decode().splitlines()
    identity_files = []
    for prefix in ("tdn/", "reference/", "scripts/", "configs/", "tests/"):
        identity_files.extend(sorted(p for p in files if p.startswith(prefix) and "__pycache__" not in Path(p).parts))
    identity_files += ["requirements.txt", "pyproject.toml", ".raven/automation.json"]
    sources = [read("outputs/"+stage+"/"+name) for stage in ("verification", "study") for name in ("source.json", "source_after.json")]
    rows = sources[0]["files"]
    source_blobs = blobs_for(identity_files+[r["path"] for r in rows])
    source_bad = [r["path"] for r in rows if sha(source_blobs[r["path"]]) != r["sha256"]]
    fingerprint = digest(rows)
    aggregate = hashlib.sha256()
    for path in identity_files:
        aggregate.update(path.encode()+b"\0"+hashlib.sha256(source_blobs[path]).digest())
    checks["source"] = dict(source_revision=REVISION, stage_files=len(rows), native_files=len(identity_files),
        stage_fingerprint=fingerprint, native_aggregate_sha256=aggregate.hexdigest(), hash_mismatches=source_bad,
        all_snapshots_equal=require("all numerical before/after source snapshots", all(s == sources[0] for s in sources)),
        fingerprints_match=require("stage source fingerprints", fingerprint == FINGERPRINT and all(s["fingerprint"] == fingerprint for s in sources)),
        immutable_files_match=require("stage files versus immutable git", not source_bad),
        aggregate_matches=require("native 659-file aggregate versus immutable git", len(identity_files) == contract["identity"]["source"]["files"] == 659 and aggregate.hexdigest() == contract["identity"]["source"]["sha256"]),
        captured_script_matches=require("captured native launch script", sha(source_blobs["scripts/response_preconditioner_regularization.sh"]) == run["provenance"]["script_sha256"]))
    contract_sha, profile_sha = sha(raw("outputs/native/contract.json")), sha(raw("outputs/native/profile.json"))
    checks["contract"] = dict(raw_contract_sha256=contract_sha, raw_profile_sha256=profile_sha,
        original_contract_matches=require("raw native contract hash", contract_sha == native["contract_sha256"]),
        original_profile_matches=require("raw native profile hash", profile_sha == contract["identity"]["profile_sha256"]),
        profile_content_matches=require("native profile content", read("outputs/native/profile.json") == contract["identity"]["profile"]),
        installed_package_count=len(contract["identity"]["software"]["packages"]))

    expected_verify = json.loads(blob(PRELAUNCH+"verification/protocol.json"))
    checks["protocols"] = {}
    for stage, command in (("verification", "verify"), ("study", "study")):
        protocol = read("outputs/"+stage+"/protocol.json")
        summary = read("outputs/"+stage+"/summary.json")
        software = read("outputs/"+stage+"/software.json")
        checks["protocols"][stage] = dict(protocol_sha256=digest(protocol), elapsed_seconds=summary["elapsed_seconds"],
            prelaunch_matches=require(stage+" immutable prelaunch protocol", protocol == dict(expected_verify, command=command)),
            contract_matches=require(stage+" native contract protocol", protocol == contract["protocols"][command]),
            summary_matches=require(stage+" summary protocol/source", digest(protocol) == summary["protocol_sha256"] and fingerprint == summary["source_fingerprint"]),
            cpu_only=require(stage+" explicit CPU/thread scope", summary["device"] == software["device"] == "cpu" and software["cuda_execution"] is False and software["threads"] == 1),
            versions_match=require(stage+" software versions", software["python"].split()[0] == contract["identity"]["software"]["python"] and all(software[k] == contract["identity"]["software"]["packages"][k] for k in ("numpy", "scipy", "torch"))),
            job_matches=require(stage+" native job ID", software["slurm_job_id"] == run["job_id"] == "2951"),
            completed_bounded=require(stage+" completed bounded audit", summary["status"] == "COMPLETED" and summary["decision"] == "BOUNDED_RETAINED_DATA_DEVELOPMENT" and summary["scientific_promotion"] is False))
    guards = native["identity_checks"]
    expected_guards = [("preflight", "before"), ("verify", "before"), ("verify", "after"), ("study", "before"), ("study", "after")]
    checks["native_guards"] = dict(records=guards,
        complete_pre_post_guards=require("all five native source/input guards", [(r["command"], r["point"]) for r in guards] == expected_guards and all(r["source_equal"] and r["input_equal"] and not r["errors"] for r in guards)),
        post_stage_receipts_match=require("persisted post-stage identity receipts", read("outputs/native/post-verify-identity.json") == guards[2] and read("outputs/native/post-study-identity.json") == guards[4]),
        chronological=require("identity check chronology", all(a["checked_at"] < b["checked_at"] for a, b in zip(guards, guards[1:]))),
        final_flags=require("final native source/input unchanged", native["source_unchanged"] is True and native["input_unchanged"] is True),
        limitation="Guard receipts record application observations; they are not external attestation of every file access.")

    checks["stage_manifests"] = {}
    stage_rows = {}
    for stage, expected_count in (("verification", 13), ("study", 6185)):
        folder = safe("outputs/"+stage)
        entries = read("outputs/"+stage+"/artifacts.json")["files"]
        bad = []
        for row in entries:
            content = raw("outputs/"+stage+"/"+row["path"])
            if len(content) != row["bytes"] or sha(content) != row["sha256"]:
                bad.append(row["path"])
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
        expected = {r["path"] for r in entries} | {"artifacts.json"}
        stage_rows[stage] = {r["path"]: r for r in entries}
        checks["stage_manifests"][stage] = dict(files=len(entries), bytes=sum(r["bytes"] for r in entries),
            hash_size_mismatches=bad, missing=sorted(expected-actual), unlisted=sorted(actual-expected),
            matches=require(stage+" complete artifact inventory", not bad and actual == expected and len(entries) == len(expected)-1 == expected_count))

    cases = list(ET.fromstring(raw("outputs/verification/verification.xml")).iter("testcase"))
    previous = list(ET.fromstring(blob(PRELAUNCH+"verification/verification.xml")).iter("testcase"))
    ids = lambda rows: [(r.attrib.get("classname"), r.attrib["name"]) for r in rows]
    actual_ids = ids(cases)
    expected_nodes = read("outputs/verification/expected_cases.json")
    expected_ids = []
    for node in expected_nodes:
        parts = node.split("::")
        expected_ids.append((parts[0].removesuffix(".py").replace("/", ".")+"".join("."+p for p in parts[1:-1]), parts[-1]))
    counts = dict(total=len(cases), expected=len(expected_nodes), failed=sum(r.find("failure") is not None for r in cases),
        errors=sum(r.find("error") is not None for r in cases), skipped=sum(r.find("skipped") is not None for r in cases),
        duplicates=len(actual_ids)-len(set(actual_ids)), exact_roster=collections.Counter(actual_ids) == collections.Counter(expected_ids))
    checks["junit"] = dict(**counts,
        all_required_passed=require("585 complete native required tests", len(cases) == 585 and counts["exact_roster"] and all(counts[k] == 0 for k in ("failed", "errors", "skipped", "duplicates"))),
        immutable_roster_matches=require("native/immutable prelaunch exact test roster", collections.Counter(actual_ids) == collections.Counter(ids(previous)) and expected_nodes == json.loads(blob(PRELAUNCH+"verification/expected_cases.json"))),
        counts_match=require("JUnit summary counts", counts == read("outputs/verification/checks.json")),
        class_counts=dict(sorted(collections.Counter(r.attrib["classname"] for r in cases).items())))
    release = read("outputs/native/verification-release.json")
    checks["verification_release"] = dict(files=len(release["hashes"]), matches=require("same-attempt verification release", release["contract_sha256"] == contract_sha and release["verification"] == "outputs/verification" and all(sha(raw("outputs/verification/"+p)) == value for p, value in release["hashes"].items())))

    p1_raw = blob(P1+"upload-index.json")
    require("immutable reviewed P1 input index hash", sha(p1_raw) == P1_INDEX_SHA)
    p1_index = json.loads(p1_raw)
    p1_files = {r["path"]: r for r in p1_index["files"]}
    inputs = read("outputs/study/inputs/input-receipt.json")
    input_identity = inputs["identity"]
    input_bad = []
    for row in inputs["files"]:
        historical = p1_files.get("outputs/study/"+row["path"])
        content = raw("outputs/study/inputs/"+row["path"])
        if historical is None or row["sha256"] != historical["sha256"] or row["bytes"] != historical["bytes"] or sha(content) != row["sha256"] or len(content) != row["bytes"]:
            input_bad.append(row["path"])
    actual_inputs = {p.relative_to(safe("outputs/study/inputs")).as_posix() for p in safe("outputs/study/inputs").rglob("*") if p.is_file()}
    expected_inputs = {r["path"] for r in inputs["files"]} | {"input-receipt.json", "access.jsonl"}
    p1_source = read("outputs/study/inputs/source.json")
    p1_protocol = read("outputs/study/inputs/protocol.json")
    checks["retained_input_bank"] = dict(files=len(inputs["files"]), bytes=sum(r["bytes"] for r in inputs["files"]), mismatches=input_bad,
        all_original_bytes_match=require("3849 consumed files equal pinned P1", len(inputs["files"]) == 3849 and not input_bad and actual_inputs == expected_inputs and inputs["retained_bytes"] == 61235622 == sum(r["bytes"] for r in inputs["files"])),
        identity_receipts_agree=require("native/contract/study/reader input identities", input_identity == contract["input_identity"] == read("outputs/native/input_identity.json") == read("outputs/study/input_identity.json")),
        pinned_archive_matches=require("exact P1 archive/index binding", input_identity["archive_sha256"] == p1_index["archive_sha256"] == P1_ARCHIVE_SHA and input_identity["index_sha256"] == P1_INDEX_SHA and input_identity["archive_bytes"] == p1_index["archive_bytes"] == 91404958 and input_identity["run_id"] == p1_index["run_id"]),
        original_source_protocol_matches=require("original numerical source/protocol identities", p1_source == read("outputs/study/inputs/source_after.json") and digest(p1_source["files"]) == input_identity["source_fingerprint"] == p1_source["fingerprint"] and digest(p1_protocol) == input_identity["protocol_sha256"]),
        no_duplicate_archive=require("old archive not recopied", inputs["original_archive_copied"] is False and not any(p.endswith(".zip") for p in actual_inputs)),
        historical_status_preserved=require("old native failed status remains visible", input_identity["original_native_status"] == "FAILED" and input_identity["numerical_study_status"] == "COMPLETED"),
        scope="Exact bytes of the previously inspected P1 bank. Input identity's numeric_arrays_loaded=false describes preparation-only verification, not this study's reads.")

    queries = read("outputs/study/inputs/query_index.json")
    parents = read("outputs/study/inputs/parents.json")
    by_query = {q["query_id"]: q for q in queries}
    train_parents = sorted(p["parent_id"] for p in parents if p["split"] == "train")
    train_ids = {q["query_id"] for q in queries if q["split"] == "train" and q["role"] == "calibration"}
    val_ids = [q["query_id"] for q in queries if q["split"] == "validation" and q["role"] == "withheld"]
    protected_ids = {q["query_id"] for q in queries if q["split"] == "diagnostic" or (q["split"] == "train" and q["role"] == "withheld")}
    withheld_ids = {q["query_id"] for q in queries if q["role"] == "withheld"}
    selection = read("outputs/study/selection.json")
    selection_sha = sha(raw("outputs/study/selection.json"))
    original_selection = read("outputs/study/inputs/selection.json")
    original_by_id = {c["model_id"]: c for c in original_selection["cells"]}
    protocol = read("outputs/study/protocol.json")
    cell_checks = []
    for cell in selection["cells"]:
        cid = cell["cell_id"]
        prefix = train_parents[:cell["training_parent_count"]]
        expected_train = [q["query_id"] for q in queries if q["parent_id"] in prefix and q["role"] == "calibration" and q["source_index"] < cell["calibration_source_count"]]
        original = original_by_id["fit-"+cid]
        expected_candidates = [("zero", None)] + [("coefficient_ridge", r) for r in protocol["coefficient_ridges"]] + [("action_energy", r) for r in protocol["action_energies"]] + [("projection", None)]
        candidates = cell["candidates"]
        chosen = {}
        for family in ("coefficient_ridge", "action_energy"):
            eligible = [c for c in candidates if c["family"] in ("zero", family) and c["eligible"]]
            winner = min(eligible, key=lambda c: c["validation_mse"])
            chosen[family] = dict(candidate=cell["selected"][family]["name"], ridge=cell["selected"][family]["ridge"],
                validation_mse=cell["selected"][family]["validation_mse"],
                validation_minimum_matches=require(cid+" "+family+" validation-only selection", cell["selected"][family] == winner and winner["eligible"]))
        cell_checks.append(dict(cell_id=cid, selected=chosen, training_queries=len(expected_train), validation_queries=len(val_ids),
            seal_matches=require(cid+" full cell seal", digest({k:v for k,v in cell.items() if k != "selection_seal"}) == cell["selection_seal"]),
            fit_file_matches=require(cid+" standalone fit record", cell == read("outputs/study/fits/"+cid+".json")),
            training_prefix_matches=require(cid+" exact nested train labels", cell["training_parent_ids"] == prefix and cell["train_query_ids"] == expected_train),
            fixed_validation_matches=require(cid+" exact validation bank", cell["validation_query_ids"] == val_ids),
            candidate_menu_matches=require(cid+" frozen candidate menu", [(c["family"], c["ridge"]) for c in candidates] == expected_candidates),
            original_model_matches=require(cid+" original P1 readout", cell["original_coefficients"] == original["selected"]["coefficients"] and cell["original_selection_seal"] == original["selection_seal"]),
            no_protected_selection_truth=require(cid+" protected truth not used", cell["diagnostic_truth_accessed"] is False and cell["training_withheld_truth_accessed"] is False)))
    checks["selection"] = dict(cells=cell_checks, selection_file_sha256=selection_sha,
        exactly_three_cells=require("three frozen training cells", [(c["training_parent_count"], c["calibration_source_count"]) for c in selection["cells"]] == [(8,8),(32,2),(32,8)]),
        six_sealed_choices=require("six complete new-run choices", selection["all_selections_sealed"] and sum(len(c["selected"]) for c in selection["cells"]) == 6 and selection["schema"] == "tdn.response_preconditioner.regularization.selection.v1"),
        explicitly_inspected=require("development and nonpromotion labels", selection["previously_inspected_development"] is True and selection["scientific_promotion"] is False))

    journal = jsonl("outputs/study/journal.jsonl")
    access = jsonl("outputs/study/inputs/access.jsonl")
    freezes = [(i,r) for i,r in enumerate(journal) if r["event"] == "all_selections_frozen"]
    freeze_i, freeze = freezes[0]
    read_events = [(i,r) for i,r in enumerate(journal) if r["event"] == "retained_query_read"]
    event_ids = [r["query_id"] for i,r in read_events]
    before_access = [r for r in access if r["protected_released"] is False]
    after_access = [r for r in access if r["protected_released"] is True]
    queries_access = [r["query_id"] for r in access if r["kind"] == "query"]
    predictions_access = [r["query_id"] for r in access if r["kind"] == "predictions"]
    evaluated = [r for r in journal if r["event"] == "query_evaluated"]
    prediction_bad = [r["query_id"] for r in evaluated if sha(raw("outputs/study/"+r["prediction_path"])) != r["prediction_sha256"]]
    checks["access_order"] = dict(journal_events=len(journal), event_counts=dict(collections.Counter(r["event"] for r in journal)),
        adapter_access_records=len(access), before_release_count=len(before_access), after_release_count=len(after_access),
        timestamps_increase=require("journal chronology", all(a["utc"] < b["utc"] for a,b in zip(journal,journal[1:]))),
        protocol_first=require("protocol frozen before read access", journal[0]["event"] == "protocol_frozen" and journal[0]["protocol_sha256"] == digest(protocol)),
        freeze_seal_matches=require("selection journal/adapter file seals", len(freezes) == 1 and freeze["selection_file_sha256"] == selection_sha and inputs["protected_release"]["selection_file_sha256"] == inputs["protected_release"]["selection_sha256"] == selection_sha and inputs["protected_release"]["selection_path"] == "selection.json"),
        all_unique_inputs_read=require("2688 unique query reads across receipts", len(event_ids) == len(set(event_ids)) == 2688 and set(event_ids) == set(by_query) and event_ids == queries_access == inputs["accessed_query_ids"]),
        all_candidates_before_freeze=require("39 candidates finished before global freeze", sum(r["event"] == "candidate_finished" for r in journal[:freeze_i]) == 39 and not any(r["event"] == "candidate_finished" for r in journal[freeze_i:])),
        training_validation_before_freeze=require("all fitting/selection inputs precede seal", {r["query_id"] for i,r in read_events if i < freeze_i} == train_ids | set(val_ids)),
        protected_truth_after_freeze=require("protected inputs after all six choices", {r["query_id"] for i,r in read_events if i > freeze_i} == protected_ids),
        adapter_gate_matches=require("adapter gate before/after ordering", len(before_access) == inputs["protected_release"]["access_count_before_release"] == 1728 and access == before_access+after_access and all(r["kind"] == "query" for r in before_access) and {r["query_id"] for r in before_access} == train_ids | set(val_ids)),
        original_predictions_after_freeze=require("all 1152 original predictions behind release", len(predictions_access) == len(set(predictions_access)) == 1152 and set(predictions_access) == withheld_ids and predictions_access == inputs["accessed_prediction_ids"] and all(r["protected_released"] for r in access if r["kind"] == "predictions")),
        evaluated_prediction_hashes_match=require("1152 new prediction hashes", len(evaluated) == len({r["query_id"] for r in evaluated}) == 1152 and not prediction_bad),
        limitation="Recorded internal read ordering is auditable, not authenticated external-access proof. Every parent remains previously inspected development evidence.")

    metrics = read("outputs/study/metrics.json")
    methods = {r["method"] for r in metrics}
    expected_methods = {kind+"-"+c["cell_id"] for c in selection["cells"] for kind in ("coefficient_ridge", "action_energy", "original", "projection")} | {"ordered_2", "ordered_8", "lawson_2", "lawson_4", "lawson_8"}
    denominators = read("outputs/study/denominators.json")
    expected_denominators = dict(attempted_candidates=39, attempted_fit_cells=3, attempted_metrics=19584,
        expected_candidates=39, expected_fit_cells=3, expected_input_queries=2688, expected_metrics=19584,
        expected_selected_models=6, failed_candidates=0, failed_metrics=0, fresh_reference_solves=0,
        ineligible_candidates=3, measured_metrics=19584, missing_metrics=0, new_timing_bundles=0,
        read_input_queries=2688, selected_models=6, unresolved_metrics=0)
    checks["denominators"] = dict(**denominators, methods=sorted(methods),
        exact_counts=require("complete expected P1R denominators", denominators == expected_denominators),
        complete_metric_slots=require("19584 exact measured query/method pairs", len(metrics) == 19584 and methods == expected_methods and {(r["query_id"],r["method"]) for r in metrics} == {(q,m) for q in withheld_ids for m in methods} and all(r["status"] == "MEASURED" for r in metrics)),
        three_projection_exclusions=require("only projection is ineligible", len([c for cell in selection["cells"] for c in cell["candidates"] if not c["eligible"]]) == 3 and all(c["eligible"] or c["family"] == "projection" for cell in selection["cells"] for c in cell["candidates"])))

    packing = read("outputs/native/packing.json")
    packed_bad = []
    for row in packing["files"]:
        content = raw("outputs/study/"+row["path"])
        if stage_rows["study"].get(row["path"]) != row or len(content) != row["bytes"] or sha(content) != row["sha256"]:
            packed_bad.append(row["path"])
    expected_packed = {p for p in stage_rows["study"] if p.startswith(("inputs/", "predictions/"))}
    index_path = args.upload_index or base.parents[1]/"upload-index.json"
    upload_raw = index_path.read_bytes()
    upload = json.loads(upload_raw)
    require("exact uploaded P1R index", sha(upload_raw) == UPLOAD_INDEX_SHA and upload["run_id"] == run["run_id"])
    omitted = upload["omitted_duplicates"]
    expected_omitted = [dict(path="outputs/native/portable-study-evidence.zip", reason="Duplicate of hash-verified original study files included in this export")]
    checks["packing"] = dict(packed_files=len(packing["files"]), packed_raw_bytes=sum(r["bytes"] for r in packing["files"]),
        original_archive_bytes=packing["archive_bytes"], archive_present=safe(packing["archive"]).exists(),
        packed_mismatches=packed_bad, elapsed_seconds=packing["elapsed_seconds"], export_omissions=omitted,
        upload_index_sha256=sha(upload_raw),
        exact_inventory=require("6155 packed files match sealed raw inputs/predictions", len(packing["files"]) == len(expected_packed) == 6155 and {r["path"] for r in packing["files"]} == expected_packed and not packed_bad),
        intentional_export_deduplication=require("only duplicate portable ZIP omitted", omitted == expected_omitted and not safe(packing["archive"]).exists() and packing["archive"] == expected_omitted[0]["path"] and packing["archive_bytes"] == 121133097),
        limitation="Export deliberately omits the redundant original ZIP after verifying every raw file. This audit validates its packing inventory, not the unavailable ZIP's byte-level identity.")

    raven = jsonl("metrics.jsonl")
    research = read("outputs/study/regularization_metrics.json")
    published = [r for r in raven if r["phase"] not in ("regularization/verify", "regularization/study")]
    expected_research = [("regularization/"+r["phase_suffix"], r["step"], r["values"]) for r in research]
    actual_research = [(r["phase"], r["step"], r["metrics"]) for r in published]
    post_study = datetime.fromisoformat(guards[-1]["checked_at"]).timestamp()
    stages = [r for r in raven if r["phase"] in ("regularization/verify", "regularization/study")]
    checks["raven_metrics"] = dict(total_rows=len(raven), research_rows=len(research), published_research_rows=len(published),
        research_by_phase=dict(collections.Counter(r["phase"].split("/")[1] for r in published)),
        exact_research_parity=require("95 research metric rows published exactly", len(research) == len(published) == 95 and expected_research == actual_research),
        finite_values=require("all published values finite", all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for r in raven for v in r["metrics"].values())),
        after_guard=require("research metrics published after post-study identity guard", all(r["t"] > post_study for r in published)),
        stage_metrics_match=require("both successful stage metrics", len(stages) == 2 and [(r["phase"],r["metrics"]["stage_exit_code"],r["metrics"]["stage_seconds"]) for r in stages] == [("regularization/"+c["command"],c["exit_code"],c["elapsed_seconds"]) for c in native["commands"]]),
        published_flag=require("research publication recorded", native["research_metrics_published"] is True))
    outputs = read(".raven/contracts/outputs.v1.json")["outputs"]
    missing_outputs = [r["path"] for r in outputs if not safe(r["path"]).is_file()]
    checks["raven_outputs"] = dict(registered=len(outputs), present=len(outputs)-len(missing_outputs),
        missing_in_deduplicated_export=missing_outputs,
        exactly_declared_omission=require("registered artifacts portable except declared duplicate", len(outputs) == len({r["path"] for r in outputs}) == 58 and all(r["required"] for r in outputs) and missing_outputs == [r["path"] for r in expected_omitted]))

    checks["telemetry"] = {}
    for prefix, name in (("outputs/native/", "worker"), ("", "supervisor")):
        index = read(prefix+"telemetry/index.json")
        samples = jsonl(prefix+"telemetry/cpu-memory.jsonl")
        coverage = index["sources"][0]["coverage"]
        partial = sum(r.get("status") == "partial" for r in samples)
        row = dict(samples=len(samples), partial_samples=partial, status=index["sources"][0]["status"],
            valid_cpu_percent_samples=sum(r.get("cpu_utilization_percent") is not None for r in samples),
            maximum_sampled_summed_rss_bytes=max(r.get("rss_bytes") or 0 for r in samples),
            maximum_sampled_cpu_percent=max((r["cpu_utilization_percent"] for r in samples if r.get("cpu_utilization_percent") is not None), default=None),
            missed_samples=coverage["missed_samples"], errors=coverage["errors"],
            partial_reasons=dict(collections.Counter(r.get("reason") for r in samples if r.get("status") == "partial")),
            counts_match=require(name+" telemetry coverage", len(samples) == coverage["samples"] and partial == coverage.get("partial_samples",0) and len(raw(prefix+"telemetry/cpu-memory.jsonl")) == coverage["bytes"]),
            timestamps_increase=require(name+" telemetry chronology", all(a["timestamp"] < b["timestamp"] for a,b in zip(samples,samples[1:]))))
        if prefix:
            with safe(prefix+"telemetry/cpu-memory.csv").open() as stream:
                rows_csv = list(csv.DictReader(stream))
            row["csv_matches"] = require("worker telemetry CSV", len(rows_csv) == len(samples) and all(all(c[k] == ("" if s[k] is None else str(s[k])) for k in c) for c,s in zip(rows_csv,samples)))
        checks["telemetry"][name] = row
    local = lambda stamp: datetime.fromisoformat(stamp).astimezone(ZoneInfo("America/Los_Angeles")).isoformat()
    checks["execution"] = dict(job_id=run["job_id"], native_state=native["status"], raven_state=run["state"],
        study_execution_status=native["study_execution_status"], native_exit_code=native["exit_code"],
        native_elapsed_seconds=native["elapsed_seconds"], study_elapsed_seconds=checks["protocols"]["study"]["elapsed_seconds"],
        created_at_utc=native["created_at"], finished_at_utc=native["finished_at"],
        created_at_America_Los_Angeles=local(native["created_at"]), finished_at_America_Los_Angeles=local(native["finished_at"]),
        scheduler_outcome=run["metadata"]["automation"]["scheduler_outcome"], independent_final_scheduler_accounting_present=False,
        startup_scheduler_state=native["allocation"]["job"]["JobState"],
        requested_cpus=int(native["allocation"]["job"]["NumCPUs"]), requested_memory=native["allocation"]["job"]["MinMemoryNode"],
        requested_wall=native["allocation"]["job"]["TimeLimit"], cpu_affinity=native["cpu_affinity"], commands=native["commands"],
        application_completed=require("native/Raven application completed", native["status"] == native["study_execution_status"] == run["state"] == "COMPLETED" and native["exit_code"] == run["metadata"]["automation"]["application_exit_code"] == 0 and all(c["exit_code"] == 0 for c in native["commands"])))
    require("native fixed resource envelope", checks["execution"]["requested_cpus"] == 4 and checks["execution"]["requested_memory"] == "8G" and checks["execution"]["requested_wall"] == "00:20:00")
    log_index = read("logs.json")["logs"]
    checks["logs"] = dict(portable=[r["path"] for r in log_index if not Path(r["path"]).is_absolute()],
        external_scheduler=[r["path"] for r in log_index if Path(r["path"]).is_absolute()])
    report = dict(schema="tdn.native-regularization-p1r-evidence-audit/v1", source_revision=REVISION,
        audit_status="PASS" if not failures else "FAIL", audit_failures=failures, checks=checks,
        scientific_promotion=False, numerical_study_status="COMPLETED", native_status_preserved="COMPLETED",
        limitations=[
            "Read-only retained-evidence audit; no uploaded code, new model fit, reference or timing experiment was executed.",
            "Final independent Slurm accounting is absent; scheduler_outcome remains unknown despite successful application/Raven lifecycle records.",
            "The archive is inspected development data. Verified internal selection ordering does not create fresh confirmation.",
            "The redundant portable ZIP was intentionally omitted by export. Its raw files and packing inventory are verified; its unavailable byte identity is not.",
            "Worker process-tree RSS is sampled, can repeat shared pages and has ten partial observations; supervisor telemetry is separate and neither is per-method peak memory.",
            "Reference uncertainty is inherited empirical evidence, not a certificate. There are no new reference or inference-timing measurements, and no GPU, speedup, novelty or scientific promotion follows.",
            "Numerical objective and prediction replay is reported separately from this provenance/access audit."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(dict(status=report["audit_status"], failures=failures, output=str(args.output)), sort_keys=True))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
