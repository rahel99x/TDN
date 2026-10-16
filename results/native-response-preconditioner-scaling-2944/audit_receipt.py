#!/usr/bin/env python3
"""Read-only P1 native job 2944 integrity audit against immutable git 7d0c620.

Uses stdlib, git and NumPy to inspect retained arrays. It executes no uploaded
code, model, fitting, teacher, or timing experiment. Evidence remains unchanged.
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
import zipfile
from zoneinfo import ZoneInfo

import numpy as np

REVISION = "7d0c6201ad3ec399568492077066bd359c59829e"
PRELAUNCH = "results/response-preconditioner-scaling-verification-20261016/"
FINGERPRINT = "53758e02afc69d1e665caed9f971ae8735ba8830f5c7487ec904f6395101e1bc"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha(canonical(value))


def selection_seal(cell):
    sealed = {k: v for k, v in cell.items() if k not in ("selection_seal", "fitting_seconds")}
    sealed["candidates"] = [{k: v for k, v in c.items() if k != "fitting_seconds"}
                            for c in cell["candidates"]]
    sealed["selected"] = {k: v for k, v in cell["selected"].items() if k != "fitting_seconds"}
    return digest(sealed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "evidence", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    root, base = args.repository.resolve(), args.evidence.resolve()
    checks, failures = {}, []

    def require(name, value):
        if not value:
            failures.append(name)
        return bool(value)

    def safe(relative):
        p = PurePosixPath(relative)
        if p.is_absolute() or ".." in p.parts or "\\" in relative:
            raise ValueError("Unsafe evidence-relative path: " + relative)
        dest = base / relative
        if not dest.resolve().is_relative_to(base) or dest.is_symlink():
            raise ValueError("Evidence path escaped root or is a symlink: " + relative)
        return dest

    def raw(path):
        return safe(path).read_bytes()

    def read(path):
        return json.loads(raw(path))

    def jsonl(path):
        return [json.loads(line) for line in raw(path).decode().splitlines()]

    def git(*parts):
        return subprocess.check_output(["git", *parts], cwd=root)

    def blob(path):
        return git("show", REVISION + ":" + path)

    native, run = read("outputs/native/status.json"), read("run.json")
    contract = read("outputs/native/contract.json")
    files = git("ls-tree", "-r", "--name-only", REVISION).decode().splitlines()
    identity_files = []
    for prefix in ("tdn/", "reference/", "scripts/", "configs/", "tests/"):
        identity_files.extend(sorted(p for p in files if p.startswith(prefix) and "__pycache__" not in Path(p).parts))
    identity_files += ["requirements.txt", "pyproject.toml", ".raven/automation.json"]
    sources = [read("outputs/" + stage + "/" + name) for stage in ("verification", "study")
               for name in ("source.json", "source_after.json")]
    rows = sources[0]["files"]
    blobs = {p: blob(p) for p in set(identity_files) | {r["path"] for r in rows}}
    bad = [r["path"] for r in rows if sha(blobs[r["path"]]) != r["sha256"]]
    fingerprint = digest(rows)
    aggregate = hashlib.sha256()
    for path in identity_files:
        aggregate.update(path.encode() + b"\0" + hashlib.sha256(blobs[path]).digest())
    checks["source"] = dict(stage_files=len(rows), native_files=len(identity_files),
        fingerprint=fingerprint, native_aggregate_sha256=aggregate.hexdigest(), mismatches_against_git=bad,
        stage_before_after_equal=require("all four numerical source snapshots equal", all(s == sources[0] for s in sources)),
        fingerprints_match=require("numerical source fingerprints", fingerprint == FINGERPRINT and all(s["fingerprint"] == fingerprint for s in sources)),
        stage_files_match_git=require("stage source hashes against immutable git", not bad),
        native_aggregate_matches=require("native source aggregate against immutable git", contract["identity"]["source"]["sha256"] == aggregate.hexdigest() and len(identity_files) == contract["identity"]["source"]["files"] == 643),
        captured_script_matches=require("captured launch script", sha(blobs["scripts/response_preconditioner_scaling.sh"]) == run["provenance"]["script_sha256"]),
        final_native_aggregate_guard_reached=False,
        limitation="source_unchanged=true was last set after verification. The pre-study aggregate check passed; packing failure prevented the final post-study aggregate check. Study source.json/source_after.json cover the numerical source independently.")
    contract_sha, profile_sha = sha(raw("outputs/native/contract.json")), sha(raw("outputs/native/profile.json"))
    checks["contract"] = dict(raw_contract_sha256=contract_sha, raw_profile_sha256=profile_sha,
        contract_hash_matches=require("original native contract bytes", contract_sha == native["contract_sha256"]),
        profile_hash_matches=require("original profile bytes", profile_sha == contract["identity"]["profile_sha256"]),
        profile_content_matches=require("profile content", read("outputs/native/profile.json") == contract["identity"]["profile"]),
        installed_package_count=len(contract["identity"]["software"]["packages"]))
    checks["protocols"] = {}
    for stage, command, previous in (("verification", "verify", "verification/protocol.json"), ("study", "study", "study-protocol.json")):
        protocol = read("outputs/" + stage + "/protocol.json")
        summary = read("outputs/" + stage + "/summary.json")
        software = read("outputs/" + stage + "/software.json")
        checks["protocols"][stage] = dict(sha256=digest(protocol), elapsed_seconds=summary["elapsed_seconds"],
            frozen_prelaunch_matches=require(stage + " prelaunch protocol", protocol == json.loads(blob(PRELAUNCH + previous))),
            contract_matches=require(stage + " contract protocol", protocol == contract["protocols"][command]),
            summary_matches=require(stage + " protocol/source summary hashes", digest(protocol) == summary["protocol_sha256"] and summary["source_fingerprint"] == fingerprint),
            cpu_only=require(stage + " explicit CPU", summary["device"] == software["device"] == "cpu" and software["cuda_execution"] is False),
            package_versions_match=require(stage + " package versions", all(software[k] == contract["identity"]["software"]["packages"][k] for k in ("numpy", "scipy", "torch"))),
            python_matches=require(stage + " Python version", software["python"].split()[0] == contract["identity"]["software"]["python"]),
            job_matches=require(stage + " job ID", software["slurm_job_id"] == run["job_id"] == "2944"),
            completed_development=require(stage + " completed bounded development", summary["status"] == "COMPLETED" and summary["decision"] == "BOUNDED_DEVELOPMENT" and summary["scientific_promotion"] is False))
    checks["stage_manifests"] = {}
    for stage in ("verification", "study"):
        folder = safe("outputs/" + stage)
        rows = read("outputs/" + stage + "/artifacts.json")["files"]
        bad = []
        for r in rows:
            content = raw("outputs/" + stage + "/" + r["path"])
            if len(content) != r["bytes"] or sha(content) != r["sha256"]:
                bad.append(r["path"])
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
        expected = {r["path"] for r in rows} | {"artifacts.json"}
        checks["stage_manifests"][stage] = dict(files=len(rows), hash_size_mismatches=bad,
            missing=sorted(expected - actual), unlisted=sorted(actual - expected),
            valid=require(stage + " full stage artifact inventory", not bad and actual == expected and len(rows) == len(expected)-1))
    require("study artifact count", checks["stage_manifests"]["study"]["files"] == 7717)

    cases = list(ET.fromstring(raw("outputs/verification/verification.xml")).iter("testcase"))
    previous = list(ET.fromstring(blob(PRELAUNCH + "verification/verification.xml")).iter("testcase"))
    ids = lambda xs: [(x.attrib.get("classname"), x.attrib["name"]) for x in xs]
    actual_ids = ids(cases)
    expected_nodes = read("outputs/verification/expected_cases.json")
    expected_ids = []
    for node in expected_nodes:
        parts = node.split("::")
        expected_ids.append((parts[0].removesuffix(".py").replace("/", ".") + "".join("." + p for p in parts[1:-1]), parts[-1]))
    counts = dict(total=len(cases), expected=len(expected_nodes), failed=sum(x.find("failure") is not None for x in cases),
        errors=sum(x.find("error") is not None for x in cases), skipped=sum(x.find("skipped") is not None for x in cases),
        duplicates=len(actual_ids)-len(set(actual_ids)), exact_roster=collections.Counter(actual_ids) == collections.Counter(expected_ids))
    checks["junit"] = dict(**counts,
        all_required_passed=require("388 exact required native checks", len(cases) == 388 and counts["exact_roster"] and all(counts[k] == 0 for k in ("failed", "errors", "skipped", "duplicates"))),
        prelaunch_roster_matches=require("native/prelaunch exact JUnit roster", collections.Counter(actual_ids) == collections.Counter(ids(previous))),
        expected_nodeids_match=require("native/prelaunch expected node IDs", expected_nodes == json.loads(blob(PRELAUNCH + "verification/expected_cases.json"))),
        checks_json_matches=require("JUnit checks.json", counts == read("outputs/verification/checks.json")),
        class_counts=dict(sorted(collections.Counter(c.attrib["classname"] for c in cases).items())))
    release = read("outputs/native/verification-release.json")
    checks["verification_release"] = dict(files=len(release["hashes"]), matches=require("same-attempt verification release hashes", release["contract_sha256"] == contract_sha and release["verification"] == "outputs/verification" and all(sha(raw("outputs/verification/" + p)) == h for p, h in release["hashes"].items())))

    protocol = read("outputs/study/protocol.json")
    parents = read("outputs/study/parents.json")
    queries = read("outputs/study/query_index.json")
    selection = read("outputs/study/selection.json")
    journal = jsonl("outputs/study/journal.jsonl")
    by_parent = {p["parent_id"]: p for p in parents}
    by_query = {q["query_id"]: q for q in queries}
    parent_bad = []
    with np.load(safe("outputs/study/parents.npz"), allow_pickle=False) as arrays:
        stored_parents = {k: arrays[k] for k in arrays.files}
    for p in parents:
        u = stored_parents[p["parent_id"]]
        field = np.ascontiguousarray(u, dtype="<f4").tobytes(order="C")
        metadata = dict(dtype="little_endian_float32", shape=list(u.shape), grid=p["grid"], lengths=p["lengths"], kappa=p["kappa"], reaction_rate=p["reaction_rate"], target="fd_same_grid")
        ok = (u.shape == (1, 1, 5, 5) and u.dtype == np.float32 and np.isfinite(u).all()
              and sha(field) == p["field_sha256"] and sha(canonical(metadata)+b"\n"+field) == p["identity_sha256"]
              and float(u.astype(np.float64).mean()) == p["actual_mean"]
              and float(u.astype(np.float64).var()) == p["actual_variance"]
              and p["seed"] == protocol["parent_seeds"][p["split"]][p["index"]]
              and p["parent_family"] == protocol["parent_families"][p["index"] % 4])
        if not ok:
            parent_bad.append(p["parent_id"])
    family_counts = {split: dict(collections.Counter(p["parent_family"] for p in parents if p["split"] == split)) for split in ("train", "validation", "diagnostic")}
    checks["parents"] = dict(count=len(parents), by_split=dict(collections.Counter(p["split"] for p in parents)),
        families_by_split=family_counts, mismatches=parent_bad,
        arrays_match=require("actual parent fields and identities", not parent_bad and set(stored_parents) == set(by_parent)),
        disjoint_identities=require("48 distinct parent physical fields, identities and seeds", len(parents) == len(by_parent) == len({p["field_sha256"] for p in parents}) == len({p["identity_sha256"] for p in parents}) == len({p["seed"] for p in parents}) == 48),
        balanced=require("balanced 32/8/8 parents across four families", all(set(family_counts[s]) == set(protocol["parent_families"]) and set(family_counts[s].values()) == {n} for s, n in (("train", 8), ("validation", 2), ("diagnostic", 2)))))
    starts = {r["query_id"]: i for i, r in enumerate(journal) if r["event"] == "query_started"}
    ends = {r["query_id"]: i for i, r in enumerate(journal) if r["event"] == "query_finished"}
    freezes = [(i, r) for i, r in enumerate(journal) if r["event"] == "all_selections_frozen"]
    freeze_i, freeze = freezes[0]
    train_parents = [p["parent_id"] for p in parents if p["split"] == "train"]
    train_ids = [q["query_id"] for q in queries if q["split"] == "train" and q["role"] == "calibration"]
    val_ids = [q["query_id"] for q in queries if q["split"] == "validation" and q["role"] == "withheld"]
    cell_checks = []
    for c in selection["cells"]:
        parent_prefix = train_parents[:c["training_parent_count"]]
        expected_train = [q["query_id"] for q in queries if q["parent_id"] in parent_prefix and q["role"] == "calibration" and q["source_index"] < c["calibration_source_count"]]
        key = c["model_id"]
        cell_checks.append(dict(model_id=key, selected_name=c["selected"]["name"], train_queries=len(expected_train), validation_queries=len(val_ids),
            seal_matches=require(key + " selection seal", selection_seal(c) == c["selection_seal"]),
            saved_fit_matches=require(key + " standalone fit", c == read("outputs/study/fits/" + key + ".json")),
            exact_training_prefix=require(key + " nested training labels", c["training_parent_ids"] == parent_prefix and c["train_query_ids"] == expected_train),
            exact_validation_bank=require(key + " fixed validation labels", c["validation_query_ids"] == val_ids),
            protected_truth_unused=require(key + " no protected truth selection", c["diagnostic_truth_accessed"] is False and c["training_withheld_truth_accessed"] is False)))
    checks["selections"] = dict(cells=cell_checks,
        nine_cells=require("nine exact scaling cells", {(c["training_parent_count"], c["calibration_source_count"]) for c in selection["cells"]} == {(p, s) for p in (8, 16, 32) for s in (2, 4, 8)} and len(selection["cells"]) == 9),
        all_sealed=require("all nine selections sealed", selection["all_selections_sealed"] and all(c["status"] == "SELECTED" for c in selection["cells"])))
    fit_events = [(i, r) for i, r in enumerate(journal) if r["event"] == "fit_cell_finished"]
    checks["access_order"] = dict(event_counts=dict(collections.Counter(r["event"] for r in journal)),
        timestamps_increase=require("journal timestamps", all(a["utc"] < b["utc"] for a, b in zip(journal, journal[1:]))),
        protocol_first=require("protocol before acquisition", journal[0]["event"] == "protocol_frozen" and journal[0]["protocol_sha256"] == digest(protocol)),
        all_queries_paired=require("2688 exact start/finish pairs", len(starts) == len(ends) == len(queries) == len(by_query) == 2688 and set(starts) == set(ends) == set(by_query) and all(starts[q] < ends[q] for q in starts)),
        selection_hash_matches=require("global selection hash", len(freezes) == 1 and freeze["selection_sha256"] == digest(selection)),
        all_fits_before_global_freeze=require("all nine fits before freeze", len(fit_events) == 9 and all(i < freeze_i for i, r in fit_events) and {r["model_id"] for i, r in fit_events} == set(freeze["model_ids"])),
        train_validation_before_freeze=require("fitting/selection labels before freeze", len(train_ids) == 1536 and len(val_ids) == 192 and all(ends[q] < freeze_i for q in train_ids + val_ids)),
        protected_truth_after_freeze=require("protected truth after all selections", all(starts[q["query_id"]] > freeze_i for q in queries if q["split"] == "diagnostic" or (q["split"] == "train" and q["role"] == "withheld"))),
        limitation="Recorded order and file seals are not authenticated proof of all external information access; validation withheld is selection data.")

    query_bad, ref_bad, source_bad = [], [], []
    max_source_rms_deviation = 0.
    for q in queries:
        content = raw("outputs/study/" + q["array_path"])
        if sha(content) != q["array_sha256"] or q != read("outputs/study/" + q["record_path"]):
            query_bad.append(q["query_id"])
        with np.load(safe("outputs/study/" + q["array_path"]), allow_pickle=False) as arrays:
            p = by_parent[q["parent_id"]]
            rms = float(np.sqrt(np.mean(arrays["source"]**2)))
            max_source_rms_deviation = max(max_source_rms_deviation, abs(rms-1.))
            if not np.array_equal(arrays["initial"], stored_parents[q["parent_id"]]) or not all(np.isfinite(arrays[k]).all() for k in arrays.files) or abs(rms-1.) > 1e-14:
                source_bad.append(q["query_id"])
        r = q["reference"]
        context_index = int(q["context_id"].rsplit("-", 1)[1])
        ok = (q["status"] == "ACCEPTED" and r["caps"] == protocol["reference"] and r["elapsed_seconds"] <= 3.
              and r["start"] == q["start"] and r["stop"] == q["stop"] and r["action_modes_attempted"] == 2 and r["source_query_count"] == 1
              and [q["start"], q["stop"]] == protocol["intervals"][context_index]
              and q["split"] == p["split"] and q["parent_family"] == p["parent_family"]
              and q["family"] == protocol[q["role"] + "_families"][q["source_index"]])
        for mode in (r, r["matched_mean"]):
            steps = mode["work"]["refinement_completed_step_counts"]
            ok = ok and (mode["accepted"] and mode["status"] == "ACCEPTED" and mode["uncertainty_is_certificate"] is False
                and mode["refinement"]["accepted"] and mode["independent"]["accepted"]
                and steps[0] == 4 and all(b == 2*a for a, b in zip(steps, steps[1:])) and steps[-1] <= 256
                and mode["work"]["independent_rhs_calls_started"] <= 4096
                and mode["independent"]["component_tolerance"] == 1e-12)
            for norm in ("rms", "maximum"):
                unc = mode["uncertainty_" + norm]
                ok = ok and (math.isfinite(unc) and 0 <= unc <= 1e-10 and unc == mode["refinement"]["uncertainty_" + norm] + mode["cross_difference"][norm])
        if not ok:
            ref_bad.append(q["query_id"])
    expected_queries = {(p["parent_id"], i, role, j) for p in parents for i in range(6)
        for role in (("calibration", "withheld") if p["split"] == "train" else ("withheld",))
        for j in range(8 if role == "calibration" else 4)}
    actual_queries = {(q["parent_id"], int(q["context_id"].rsplit("-", 1)[1]), q["role"], q["source_index"]) for q in queries}
    checks["references"] = dict(query_count=len(queries), context_count=len({q["context_id"] for q in queries}),
        queries_by_split_role=dict(collections.Counter(q["split"] + "/" + q["role"] for q in queries)),
        hash_record_mismatches=query_bad, reference_acceptance_mismatches=ref_bad, array_mismatches=source_bad,
        complete_expected_queries=require("all planned 2688 query slots", expected_queries == actual_queries and len(expected_queries) == len(queries)),
        arrays_and_records_match=require("query arrays/records/index parity", not query_bad and not source_bad),
        caps_and_acceptance_match=require("all U/G references accepted within frozen caps", not ref_bad),
        reference_index_matches=require("reference index parity", read("outputs/study/references.json") == [dict(query_id=q["query_id"], reference=q["reference"]) for q in queries]),
        maximum_reference_elapsed_seconds=max(q["reference"]["elapsed_seconds"] for q in queries),
        maximum_source_rms_deviation=max_source_rms_deviation,
        maximum_uncertainty_rms=max(m["uncertainty_rms"] for q in queries for m in (q["reference"], q["reference"]["matched_mean"])),
        maximum_uncertainty_maximum=max(m["uncertainty_maximum"] for q in queries for m in (q["reference"], q["reference"]["matched_mean"])),
        U_final_steps=dict(collections.Counter(q["reference"]["work"]["refinement_completed_step_counts"][-1] for q in queries)),
        G_final_steps=dict(collections.Counter(q["reference"]["matched_mean"]["work"]["refinement_completed_step_counts"][-1] for q in queries)),
        uncertainty_is_certificate=False)
    metrics, timing = read("outputs/study/metrics.json"), read("outputs/study/timing_records.json")
    methods = {c["model_id"] for c in selection["cells"]} | {"ordered_2", "ordered_8", "lawson_2", "lawson_4", "lawson_8"}
    expected_pairs = {(q["query_id"], m) for q in queries if q["role"] == "withheld" for m in methods}
    checks["denominators"] = dict(reference=read("outputs/study/denominators.json"), evaluation=read("outputs/study/evaluation_denominators.json"),
        complete_metrics=require("all 16128 measured query-method slots", len(metrics) == len(expected_pairs) == 16128 and {(m["query_id"], m["method"]) for m in metrics} == expected_pairs and all(m["status"] == "MEASURED" for m in metrics)),
        complete_timings=require("1008 successful timing bundles", len(timing) == 1008 and all(t["status"] == "PASS" and math.isfinite(t["seconds"]) and t["seconds"] >= 0 for t in timing)),
        source_calls=4*len(timing))
    require("reference denominator exact", checks["denominators"]["reference"] == dict(accepted=2688, attempted=2688, descendants_are_independent=False, failed=0, independent_parent_count=48, missing=0, parents_by_split=dict(diagnostic=8, train=32, validation=8), planned=2688, unresolved=0))
    require("evaluation denominator exact", checks["denominators"]["evaluation"] == dict(attempted_metrics=16128, attempted_timing_bundles=1008, expected_fit_cells=9, expected_metrics=16128, expected_timing_bundles=1008, failed_metrics=0, failed_timing_bundles=0, measured_metrics=16128, missing_metrics=0, missing_timing_bundles=0, selected_fit_cells=9, unresolved_metrics=0))
    context_timing = read("outputs/study/timing.json")
    timing_methods = {"fit-p032-s08", "ordered_2", "ordered_8", "lawson_2", "lawson_4", "lawson_8"}
    expected_pattern = collections.Counter((phase, rnd, method) for phase, rounds in (("cold", 1), ("warmup", 1), ("paired", 5)) for rnd in range(rounds) for method in timing_methods)
    expected_contexts = {p["parent_id"] + "-context-" + str(i).zfill(2) for p in parents if p["split"] == "diagnostic" and p["index"] < 4 for i in range(6)}
    flattened = [dict(context_id=c["context_id"], parent_id=c["parent_id"], parent_family=c["parent_family"], **r) for c in context_timing for r in c["records"]]
    timing_journal = [{k: v for k, v in r.items() if k not in ("event", "utc")} for r in journal if r["event"] == "timing_observation"]
    timing_journal_projection = [{k: v for k, v in r.items() if k not in ("parent_id", "parent_family")} for r in timing]
    checks["timing_scope"] = dict(contexts=len(context_timing), parents=4, methods=sorted(timing_methods), bundles=len(timing), source_calls=4*len(timing),
        selected_cell="fit-p032-s08", randomized_paired_rounds=5, cold_bundles=144, warmup_bundles=144, paired_bundles=720,
        context_coverage_matches=require("timing first diagnostic parent of each family/all six contexts", len(context_timing) == 24 and {c["context_id"] for c in context_timing} == expected_contexts),
        round_method_coverage_matches=require("exact cold/warmup/five paired timing pattern", all(collections.Counter((r["phase"], r["round"], r["method"]) for r in c["records"]) == expected_pattern for c in context_timing)),
        retained_and_journal_match=require("timing raw/context/journal parity", flattened == timing and timing_journal_projection == timing_journal),
        scope="Every bundle sequentially evaluates four withheld sources with independent fresh setup for each. First benchmark invocation is not process cold start; fitting/reference acquisition is offline and excluded from these action timings.")

    with zipfile.ZipFile(safe("outputs/native/portable-study-evidence.zip")) as bundle:
        names = bundle.namelist()
        packing_raw = bundle.read("packing-manifest.json")
        packing = json.loads(packing_raw)
        packed_bad = []
        for row in packing["files"]:
            content = bundle.read(row["path"])
            if len(content) != row["bytes"] or sha(content) != row["sha256"] or content != raw("outputs/study/" + row["path"]):
                packed_bad.append(row["path"])
        expected_names = {r["path"] for r in packing["files"]} | {"packing-manifest.json"}
    compact_manifest_bytes = len(canonical(packing))
    writer_lower_bound_bytes = compact_manifest_bytes + 1
    native_code = blob("scripts/response_preconditioner_scaling_native.py").decode()
    writer_code = blob("scripts/response_diagnostic_raven.py").decode()
    raven_metrics = jsonl("metrics.jsonl")
    checks["packing_failure"] = dict(archive_bytes=safe("outputs/native/portable-study-evidence.zip").stat().st_size,
        packed_files=len(packing["files"]), embedded_manifest_bytes=len(packing_raw), embedded_manifest_compact_bytes=compact_manifest_bytes,
        embedded_manifest_compact_with_newline_bytes=writer_lower_bound_bytes,
        coordinator_write_limit_bytes=1 << 20, attempted_final_manifest_lower_bound_bytes=writer_lower_bound_bytes,
        packed_mismatches=packed_bad,
        embedded_matches_raw=require("7680 packed files match raw sealed study", not packed_bad and len(packing["files"]) == 7680 and set(names) == expected_names and len(names) == len(expected_names)),
        packing_json_absent=require("failed packing metadata absent", not safe("outputs/native/packing.json").exists()),
        lower_bound_exceeds_limit=require("retained manifest exceeds original 1 MiB limit", writer_lower_bound_bytes > 1 << 20),
        original_write_has_default_limit=require("immutable source confirms small metadata writer", 'write_json(native / "packing.json", manifest, root=ROOT)' in native_code and 'def write_json(path, value, *, root):' in writer_code and 'if len(raw) > SMALL_BYTES:' in writer_code and 'SMALL_BYTES = 1 << 20' in writer_code),
        error_retained=require("native/Raven packing failure retained", native["status"] == run["state"] == "FAILED" and native["exit_code"] == run["metadata"]["automation"]["application_exit_code"] == 2 and native["error"] == "ValueError: Retained coordinator metadata exceeds its small-file budget"),
        scientific_children_success=require("verification and study children exited zero", len(native["commands"]) == 2 and all(c["exit_code"] == 0 for c in native["commands"])),
        raven_metric_rows=len(raven_metrics), raven_metric_phases=[r["phase"] for r in raven_metrics],
        study_series_unpublished=require("Raven contains verification-only metrics", len(raven_metrics) == 1 and raven_metrics[0]["phase"] == "scaling/verify"),
        diagnosis="Numerical study completed; the closed portable ZIP survived. Writing its enlarged coordinator metadata failed before study artifact registration/metric publication and the final native source check. Do not rewrite the failed status or resubmit science to recover this export.")

    checks["telemetry"] = {}
    for prefix, name in (("outputs/native/", "worker"), ("", "supervisor")):
        index = read(prefix + "telemetry/index.json")
        samples = jsonl(prefix + "telemetry/cpu-memory.jsonl")
        coverage = index["sources"][0]["coverage"]
        partial = sum(s.get("status") == "partial" for s in samples)
        row = dict(samples=len(samples), partial_samples=partial, status=index["sources"][0]["status"],
            missed_samples=coverage["missed_samples"], errors=coverage["errors"],
            valid_cpu_percent_samples=sum(s.get("cpu_utilization_percent") is not None for s in samples),
            maximum_sampled_summed_rss_bytes=max(s.get("rss_bytes") or 0 for s in samples),
            maximum_sampled_cpu_percent=max((s["cpu_utilization_percent"] for s in samples if s.get("cpu_utilization_percent") is not None), default=None),
            first_timestamp=samples[0]["timestamp"], last_timestamp=samples[-1]["timestamp"],
            partial_reasons=dict(collections.Counter(s.get("reason") for s in samples if s.get("status") == "partial")),
            coverage_matches=require(name + " telemetry counts/bytes", len(samples) == coverage["samples"] and partial == coverage.get("partial_samples", 0) and len(raw(prefix + "telemetry/cpu-memory.jsonl")) == coverage["bytes"]),
            timestamps_increase=require(name + " telemetry timestamps", all(a["timestamp"] < b["timestamp"] for a, b in zip(samples, samples[1:]))))
        if prefix:
            with safe(prefix + "telemetry/cpu-memory.csv").open() as stream:
                csv_samples = list(csv.DictReader(stream))
            row["csv_matches"] = require("worker telemetry CSV", len(csv_samples) == len(samples) and all(all(c[k] == ("" if s[k] is None else str(s[k])) for k in c) for c, s in zip(csv_samples, samples)))
        checks["telemetry"][name] = row
    require("worker partial samples preserved", checks["telemetry"]["worker"]["samples"] == 485 and checks["telemetry"]["worker"]["partial_samples"] == 8)
    outputs = read(".raven/contracts/outputs.v1.json")["outputs"]
    paths = [r["path"] for r in outputs]
    checks["raven_outputs"] = dict(registered_files=len(outputs),
        all_present=require("registered Raven outputs present", len(paths) == len(set(paths)) and all(r["required"] and safe(r["path"]).is_file() for r in outputs)),
        study_registration_count=sum(p.startswith("outputs/study/") for p in paths),
        note="Bulk and raw study evidence is retained in the export even though packing failure prevented study registration.")
    local = lambda s: datetime.fromisoformat(s).astimezone(ZoneInfo("America/Los_Angeles")).isoformat()
    checks["execution"] = dict(job_id=run["job_id"], raven_state=run["state"], native_state=native["status"], native_exit_code=native["exit_code"],
        scheduler_outcome=run["metadata"]["automation"]["scheduler_outcome"], independent_final_scheduler_accounting_present=False,
        startup_scheduler_state=native["allocation"]["job"]["JobState"],
        requested_cpus=int(native["allocation"]["job"]["NumCPUs"]), requested_memory=native["allocation"]["job"]["MinMemoryNode"],
        requested_wall=native["allocation"]["job"]["TimeLimit"], affinity=native["cpu_affinity"],
        created_at_utc=native["created_at"], finished_at_utc=native["finished_at"],
        created_at_America_Los_Angeles=local(native["created_at"]), finished_at_America_Los_Angeles=local(native["finished_at"]),
        native_elapsed_seconds=native["elapsed_seconds"], study_elapsed_seconds=checks["protocols"]["study"]["elapsed_seconds"], commands=native["commands"])
    require("native resource envelope", checks["execution"]["requested_cpus"] == 4 and checks["execution"]["requested_memory"] == "8G" and checks["execution"]["requested_wall"] == "00:20:00")
    report = dict(schema="tdn.native-scaling-p1-evidence-audit/v1", source_revision=REVISION,
        audit_status="PASS" if not failures else "FAIL", audit_failures=failures, checks=checks,
        scientific_promotion=False, native_status_preserved="FAILED", numerical_study_status="COMPLETED",
        limitations=[
            "This audit verifies retained evidence, not a new scientific run or independent teacher recomputation.",
            "No final independent Slurm accounting is present; scheduler_outcome remains unknown.",
            "The native aggregate source guard after study was not reached; numerical before/after stage snapshots match immutable git.",
            "Raven published only verification metrics because study evidence packing failed; raw scientific evidence remains available.",
            "Worker process-tree RSS is sampled, includes potentially repeated shared pages, and has eight partial samples. It is not per-method peak memory.",
            "These are inspected development parents, including prefixes exercised by implementation verification, on a 5x5 grid. No confirmation, rollout, GPU, population learning-law or speedup claim follows.",
            "Reference refinement/independent agreement is an uncertainty estimate, not a rigorous certificate.",
            "Numerical accuracy, fitting objectives and cost comparisons are audited in separate retained analyses."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(dict(status=report["audit_status"], failures=failures, output=str(args.output)), sort_keys=True))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
