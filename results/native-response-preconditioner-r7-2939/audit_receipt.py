#!/usr/bin/env python3
"""Read-only R7 job 2939 audit; stdlib/git only, no uploaded code execution.

Run with --repository /path/to/TDN --evidence /path/to/extracted/run
and --output /path/to/new/integrity-review.json. Immutable git source is used;
no model, teacher, fitting, or timing experiments are executed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import xml.etree.ElementTree as ET

REVISION = "43837c5ee264f7deed33c64a6e7c0cd553b82938"
PRELAUNCH = "results/response-preconditioner-r7-verification-20261015/"
FINGERPRINT = "abe17b47d43b50d0ca8fb797a4faf76dce8307898ff60835d00c0f985061df9c"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "evidence", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    root, base = args.repository.resolve(), args.evidence.resolve()
    report = dict(schema="tdn.native-r7-evidence-audit/v1", source_revision=REVISION,
                  scientific_promotion=False, checks={}, limitations=[])
    checks = report["checks"]
    failures = []

    def require(name, condition):
        if not condition:
            failures.append(name)
        return bool(condition)

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
    source_bad = [r["path"] for r in rows if sha(blobs[r["path"]]) != r["sha256"]]
    fingerprint = digest(rows)
    aggregate = hashlib.sha256()
    for path in identity_files:
        aggregate.update(path.encode() + b"\0" + hashlib.sha256(blobs[path]).digest())
    checks["source"] = dict(stage_files=len(rows), mismatches_against_git=source_bad,
        all_stage_before_after_equal=require("stage sources before/after equal", all(s == sources[0] for s in sources)),
        fingerprint=fingerprint, fingerprint_matches=require("stage source fingerprints", fingerprint == FINGERPRINT and all(s["fingerprint"] == fingerprint for s in sources)),
        all_stage_files_match_git=require("stage source files", not source_bad),
        native_files=len(identity_files), native_sha256=aggregate.hexdigest(),
        native_aggregate_matches=require("native source aggregate", contract["identity"]["source"]["sha256"] == aggregate.hexdigest() and contract["identity"]["source"]["files"] == len(identity_files)),
        script_hash_matches=require("captured launch script", sha(blobs["scripts/response_preconditioner_r7.sh"]) == run["provenance"]["script_sha256"]))
    contract_sha = sha(raw("outputs/native/contract.json"))
    profile_sha = sha(raw("outputs/native/profile.json"))
    checks["contract"] = dict(raw_contract_sha256=contract_sha, raw_profile_sha256=profile_sha,
        original_bytes_preserved=True,
        contract_hash_matches=require("native raw contract hash", contract_sha == native["contract_sha256"]),
        profile_hash_matches=require("native raw profile hash", profile_sha == contract["identity"]["profile_sha256"]),
        profile_matches=require("native profile content", read("outputs/native/profile.json") == contract["identity"]["profile"]),
        installed_package_count=len(contract["identity"]["software"]["packages"]))

    checks["protocols"] = {}
    for stage, command, prelaunch in (("verification", "verify", "verification/protocol.json"), ("study", "study", "study-protocol.json")):
        protocol = read("outputs/" + stage + "/protocol.json")
        summary = read("outputs/" + stage + "/summary.json")
        software = read("outputs/" + stage + "/software.json")
        checks["protocols"][stage] = dict(sha256=digest(protocol),
            matches_prelaunch=require(stage + " frozen prelaunch protocol", protocol == json.loads(blob(PRELAUNCH + prelaunch))),
            matches_contract=require(stage + " contract protocol", protocol == contract["protocols"][command]),
            summary_hash_matches=require(stage + " protocol summary hash", digest(protocol) == summary["protocol_sha256"]),
            source_matches=require(stage + " summary source", summary["source_fingerprint"] == fingerprint),
            cpu_only=require(stage + " explicit CPU", summary["device"] == software["device"] == "cpu" and software["cuda_execution"] is False),
            package_versions_match=require(stage + " package versions", all(software[k] == contract["identity"]["software"]["packages"][k] for k in ("numpy", "scipy", "torch"))),
            python_version_matches=require(stage + " Python version", software["python"].split()[0] == contract["identity"]["software"]["python"]),
            job_matches=require(stage + " job identity", software["slurm_job_id"] == run["job_id"] == "2939"),
            bounded_development=require(stage + " no scientific promotion", summary["decision"] == "BOUNDED_DEVELOPMENT" and summary["scientific_promotion"] is False),
            completed=require(stage + " completed", summary["status"] == "COMPLETED"))

    checks["stage_manifests"] = {}
    for stage in ("verification", "study"):
        folder = base / "outputs" / stage
        rows = read("outputs/" + stage + "/artifacts.json")["files"]
        bad = [r["path"] for r in rows if len(raw("outputs/" + stage + "/" + r["path"])) != r["bytes"] or sha(raw("outputs/" + stage + "/" + r["path"])) != r["sha256"]]
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
        expected = {r["path"] for r in rows} | {"artifacts.json"}
        checks["stage_manifests"][stage] = dict(files=len(rows), hash_size_mismatches=bad,
            missing=sorted(expected - actual), unlisted=sorted(actual - expected),
            valid=require(stage + " artifact manifest", not bad and actual == expected and len(rows) == len(expected) - 1))

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
                  duplicates=len(actual_ids) - len(set(actual_ids)), exact_roster=collections.Counter(actual_ids) == collections.Counter(expected_ids))
    checks["junit"] = dict(**counts,
        all_required_passed=require("298 complete JUnit passes", len(cases) == 298 and counts["exact_roster"] and all(counts[k] == 0 for k in ("failed", "errors", "skipped", "duplicates"))),
        prelaunch_roster_matches=require("prelaunch JUnit roster", collections.Counter(actual_ids) == collections.Counter(ids(previous))),
        expected_nodeids_match=require("prelaunch expected node IDs", expected_nodes == json.loads(blob(PRELAUNCH + "verification/expected_cases.json"))),
        checks_json_matches=require("JUnit checks.json", counts == read("outputs/verification/checks.json")),
        class_counts=dict(sorted(collections.Counter(c.attrib["classname"] for c in cases).items())))
    release = read("outputs/native/verification-release.json")
    checks["verification_release"] = dict(files=len(release["hashes"]),
        matches=require("verification release hashes", release["contract_sha256"] == contract_sha and release["verification"] == "outputs/verification" and all(sha(raw("outputs/verification/" + p)) == h for p, h in release["hashes"].items())))

    journal = jsonl("outputs/study/journal.jsonl")
    starts = {r["query_id"]: i for i, r in enumerate(journal) if r["event"] == "query_started"}
    ends = {r["query_id"]: i for i, r in enumerate(journal) if r["event"] == "query_finished"}
    seals = {r["context_id"]: (i, r) for i, r in enumerate(journal) if r["event"] == "local_coefficients_sealed"}
    selection_events = [(i, r) for i, r in enumerate(journal) if r["event"] == "selection_frozen"]
    selection_i, selection_event = selection_events[0]
    selection = read("outputs/study/selection.json")
    queries = read("outputs/study/query_index.json")
    metrics = read("outputs/study/metrics.json")
    references = read("outputs/study/references.json")
    timing = read("outputs/study/timing_records.json")
    context_timing = read("outputs/study/timing.json")
    contexts = sorted({r["context_id"] for r in queries})
    train_ids = [r["query_id"] for r in queries if r["split"] == "train" and r["role"] == "calibration"]
    val_ids = [r["query_id"] for r in queries if r["split"] == "validation" and r["role"] == "withheld"]
    query_bad = [r["query_id"] for r in queries if sha(raw("outputs/study/" + r["array_path"])) != r["array_sha256"]]
    checks["access_order"] = dict(journal_events=len(journal), event_counts=dict(collections.Counter(r["event"] for r in journal)),
        timestamps_increasing=require("journal timestamp order", all(a["utc"] < b["utc"] for a, b in zip(journal, journal[1:]))),
        protocol_first=require("protocol frozen before labels", journal[0]["event"] == "protocol_frozen" and journal[0]["protocol_sha256"] == checks["protocols"]["study"]["sha256"]),
        selection_hash_matches=require("selection seal", len(selection_events) == 1 and selection_event["selection_sha256"] == digest(selection)),
        fit_labels_exact=require("48 train calibration labels", selection["train_query_ids"] == train_ids and len(train_ids) == 48),
        selection_labels_exact=require("24 validation withheld labels", selection["validation_query_ids"] == val_ids and len(val_ids) == 24),
        fitting_labels_before_seal=require("fitting labels before global seal", all(ends[q] < selection_i for q in train_ids + val_ids)),
        diagnostic_after_global_seal=require("diagnostic labels after global seal", all(starts[r["query_id"]] > selection_i for r in queries if r["split"] == "diagnostic")),
        training_withheld_after_global_seal=require("training withheld after global seal", all(starts[r["query_id"]] > selection_i for r in queries if r["split"] == "train" and r["role"] == "withheld")),
        all_query_start_finish_paired=require("query journal pairs", len(starts) == len(ends) == 192 and set(starts) == set(ends) and all(starts[k] < ends[k] for k in starts)),
        local_fit_hashes_match=require("24 local fit seals", len(seals) == 24 and all(r["fit_sha256"] == digest(read("outputs/study/fits/" + cid + ".json")) for cid, (_, r) in seals.items())),
        local_calibration_before_seal=require("local calibration before fit seal", all(ends[r["query_id"]] < seals[r["context_id"]][0] for r in queries if r["role"] == "calibration")),
        diagnostic_withheld_after_local_seal=require("diagnostic withheld after local seal", all(starts[r["query_id"]] > seals[r["context_id"]][0] for r in queries if r["split"] == "diagnostic" and r["role"] == "withheld")),
        limitation="Recorded ordering and file seals, not authenticated proof of external information access; validation withheld is selection data.")
    methods = {r["method"] for r in metrics}
    query_ids = {r["query_id"] for r in queries}
    metric_pairs = [(r["query_id"], r["method"]) for r in metrics]
    flattened = [dict(context_id=c["context_id"], **r) for c in context_timing for r in c["records"]]
    timing_journal = [{k: v for k, v in r.items() if k not in ("event", "utc")} for r in journal if r["event"] == "timing_observation"]
    timing_pattern = collections.Counter((r["phase"], r["round"], r["method"]) for r in context_timing[0]["records"])
    expected_pattern = collections.Counter((phase, rnd, method) for phase, rounds in (("cold", 1), ("warmup", 1), ("paired", 5)) for rnd in range(rounds) for method in context_timing[0]["methods"])
    checks["denominators"] = dict(queries=len(queries), contexts=len(contexts), parents=len({r["parent_id"] for r in queries}),
        metric_rows=len(metrics), methods=sorted(methods), timing_bundles=len(timing), timed_source_calls=len(timing) * 4,
        query_array_hash_mismatches=query_bad, query_array_hashes_valid=require("192 retained query array hashes", not query_bad),
        complete_query_bank=require("192 source queries", len(queries) == len(query_ids) == 192 and len(contexts) == 24 and all(r["status"] == "ACCEPTED" and r["reference"]["accepted"] and r["reference"]["matched_mean"]["accepted"] for r in queries)),
        reference_index_matches=require("reference index", references == [dict(query_id=r["query_id"], reference=r["reference"]) for r in queries]),
        complete_metric_bank=require("2112 query-method rows", len(metrics) == len(set(metric_pairs)) == 2112 and set(metric_pairs) == {(q, m) for q in query_ids for m in methods} and all(r["status"] == "MEASURED" for r in metrics)),
        complete_timing_bank=require("336 timing bundles", len(timing) == 336 and flattened == timing == timing_journal and all(r["status"] == "PASS" and math.isfinite(r["seconds"]) and r["seconds"] >= 0 for r in timing)),
        timing_pattern_matches=require("cold/warmup/five paired rounds", timing_pattern == expected_pattern and all(collections.Counter((r["phase"], r["round"], r["method"]) for r in c["records"]) == expected_pattern for c in context_timing)),
        reference_denominators=read("outputs/study/denominators.json"), evaluation_denominators=read("outputs/study/evaluation_denominators.json"))
    require("reference denominators exact", checks["denominators"]["reference_denominators"] == dict(accepted=192, attempted=192, descendants_are_independent=False, failed=0, independent_parent_count=4, missing=0, planned=192, unresolved=0))
    require("evaluation denominators exact", checks["denominators"]["evaluation_denominators"] == dict(attempted_metrics=2112, attempted_timing_bundles=336, expected_metrics=2112, expected_timing_bundles=336, failed_metrics=0, failed_timing_bundles=0, measured_metrics=2112, missing_metrics=0, missing_timing_bundles=0, unresolved_metrics=0))

    outputs = read(".raven/contracts/outputs.v1.json")["outputs"]
    paths = [r["path"] for r in outputs]
    checks["raven_outputs"] = dict(required_files=len(outputs),
        all_present_relative_required=require("Raven portable outputs", len(paths) == len(set(paths)) and all(r["required"] and safe(r["path"]).is_file() for r in outputs)))
    raven = jsonl("metrics.jsonl")
    science = [r for r in raven if r["phase"].startswith(("r7/train/", "r7/validation/", "r7/diagnostic/"))]
    rv_timing = [r for r in raven if r["phase"].startswith("r7/timing/")]
    keys = ("rms", "maximum", "mean_error", "centered_rms", "start", "stop", "reference_uncertainty_rms", "reference_uncertainty_maximum", "correction_rms", "correction_relative_rms", "elapsed_seconds", "U_minus_P_rms", "U_minus_G_rms", "U_minus_P_uncertainty_rms", "U_minus_G_uncertainty_rms", "U_minus_P_relative_error", "U_minus_G_relative_error", "K_minus_P_rms")
    science_bad = []
    for i, (a, b) in enumerate(zip(metrics, science)):
        expected = {k: a[k] for k in keys if a.get(k) is not None}
        expected["reference_accepted"] = float(a["reference_accepted"])
        phase = "r7/" + "/".join(a[k] for k in ("split", "role", "family", "method"))
        if expected != b["metrics"] or phase != b["phase"] or i != b["step"]:
            science_bad.append(i)
    timing_bad = []
    timed_contexts = list(dict.fromkeys(r["context_id"] for r in timing))
    for i, (a, b) in enumerate(zip(timing, rv_timing)):
        expected = dict(context_index=timed_contexts.index(a["context_id"]), success=int(a["status"] == "PASS"), bundle_seconds=a["seconds"], round=a["round"], position=a["position"])
        if expected != b["metrics"] or b["phase"] != "r7/timing/" + a["phase"] + "/" + a["method"] or b["step"] != i:
            timing_bad.append(i)
    with safe("outputs/study/metrics.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    csv_bad = [i for i, (a, b) in enumerate(zip(metrics, csv_rows)) if any(b[k] != ("" if a.get(k) is None else str(a[k])) for k in b)]
    checks["raven_metrics"] = dict(raw_science_rows=len(metrics), raven_science_rows=len(science), raven_timing_rows=len(rv_timing), total_raven_rows=len(raven),
        science_mismatches=science_bad, timing_mismatches=timing_bad, csv_mismatches=csv_bad,
        science_parity=require("Raven science row parity", len(metrics) == len(science) == len(csv_rows) and not science_bad and not csv_bad),
        timing_parity=require("Raven timing row parity", len(rv_timing) == len(timing) and not timing_bad),
        context_identity="Science series include split/role/family/method, but parent/context identity remains in raw artifacts. Timing has context_index.")

    checks["telemetry"] = {}
    for prefix, name in (("outputs/native/", "worker"), ("", "supervisor")):
        index = read(prefix + "telemetry/index.json")
        samples = jsonl(prefix + "telemetry/cpu-memory.jsonl")
        coverage = index["sources"][0]["coverage"]
        partial = sum(s.get("status") == "partial" for s in samples)
        row = dict(samples=len(samples), partial_samples=partial, valid_cpu_percent_samples=sum(s.get("cpu_utilization_percent") is not None for s in samples),
            missed_samples=coverage["missed_samples"], errors=coverage["errors"], status=index["sources"][0]["status"],
            maximum_sampled_summed_rss_bytes=max(s.get("rss_bytes") or 0 for s in samples),
            partial_reasons=dict(collections.Counter(s.get("reason") for s in samples if s.get("status") == "partial")),
            counts_match=require(name + " telemetry coverage", len(samples) == coverage["samples"] and partial == coverage.get("partial_samples", 0) and len(raw(prefix + "telemetry/cpu-memory.jsonl")) == coverage["bytes"]),
            timestamp_increasing=require(name + " telemetry timestamps", all(a["timestamp"] < b["timestamp"] for a, b in zip(samples, samples[1:]))))
        if prefix:
            with safe(prefix + "telemetry/cpu-memory.csv").open() as handle:
                csv_samples = list(csv.DictReader(handle))
            row["csv_matches"] = require("worker telemetry CSV", len(csv_samples) == len(samples) and all(all(c[k] == ("" if s[k] is None else str(s[k])) for k in c) for c, s in zip(csv_samples, samples)))
        checks["telemetry"][name] = row
    checks["execution"] = dict(job_id=run["job_id"], raven_state=run["state"], application_exit_code=run["metadata"]["automation"]["application_exit_code"],
        native_state=native["status"], native_exit_code=native["exit_code"], elapsed_seconds=native["elapsed_seconds"], created_at=native["created_at"], finished_at=native["finished_at"],
        scheduler_outcome=run["metadata"]["automation"]["scheduler_outcome"], startup_job_state=native["allocation"]["job"]["JobState"],
        source_unchanged=native["source_unchanged"], requested_cpus=4, requested_memory_gib=8, requested_wall_minutes=20,
        successful_application=require("native/Raven application success", run["state"] == native["status"] == "COMPLETED" and native["exit_code"] == run["metadata"]["automation"]["application_exit_code"] == 0 and native["source_unchanged"] is True),
        commands=native["commands"])
    log_paths = [r["path"] for r in read("logs.json")["logs"]]
    checks["logs"] = dict(portable=[p for p in log_paths if not Path(p).is_absolute()], external_scheduler=[p for p in log_paths if Path(p).is_absolute()],
        outer_stdout_bytes=len(raw("logs/stdout.log")), outer_stderr_bytes=len(raw("logs/stderr.log")))
    report["limitations"] = [
        "This audits retained bytes and committed source correspondence; file hashes and journals are not authenticated provenance or proof of outside information access.",
        "Independent final Slurm sacct/scontrol accounting is absent. Raven application/native completion and exit 0 are present; scheduler_outcome is unknown and RUNNING allocation fields are startup snapshots.",
        "Original raw prepared contract and profile bytes are preserved and their hashes verified. The full installed package inventory is recorded, but installed binary/package bytes were not supplied.",
        "Four stage logs are portable. Scheduler stdout/stderr remain absolute host paths; empty outer stdout/stderr are not copies of those streams.",
        "Worker telemetry measures sampled local-process-tree summed RSS, which can repeat shared pages and omit departed processes; it is not allocation peak or per-method peak. CPU intervals affected by process churn are unavailable, not zero. Supervisor telemetry is separate. No GPU execution is inferred.",
        "The 192 accepted references use an empirical refinement/independent-agreement policy, not a certified error bound. Four reused development parents and their source/time descendants are not fresh independent confirmation.",
        "Scientific numerical replay is separate from this integrity audit. Successful execution and complete denominators do not establish architecture utility, matched-tolerance speedup, novelty, or historical gate promotion."]
    report["status"] = "PASS_WITH_LIMITATIONS" if not failures else "FAIL"
    report["failed_integrity_checks"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(status=report["status"], failures=failures, output=str(args.output))))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
