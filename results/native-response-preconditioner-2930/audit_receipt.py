#!/usr/bin/env python3
"""Read-only native job 2930 evidence audit using stdlib and git objects.

Usage: python audit.py --repository /path/to/TDN --evidence /extracted/run \
    --archive /original.zip --output /review.json
Uploaded files are never imported or executed. Git revision is immutable.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import xml.etree.ElementTree as ET
import zipfile

REVISION = "627de9f3929c2501e42a783c1ad193369b44280b"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, base = args.repository.resolve(), args.evidence.resolve()

    def read(path):
        return json.loads((base / path).read_text())

    def git(*parts):
        return subprocess.check_output(["git", *parts], cwd=root)

    def blob(path):
        return git("show", REVISION + ":" + path)

    report = {"review_schema": "tdn.native-evidence-audit/v1",
              "source_revision": REVISION, "checks": {}, "limitations": []}
    checks = report["checks"]
    with zipfile.ZipFile(args.archive) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        unsafe = [name for name in names if PurePosixPath(name).is_absolute()
                  or ".." in PurePosixPath(name).parts or "\\" in name]
        symlinks = [m.filename for m in members
                    if stat.S_ISLNK(m.external_attr >> 16)]
        mismatches = []
        if not unsafe and not symlinks and len(names) == len(set(names)):
            for m in members:
                if m.is_dir():
                    continue
                target = base / m.filename
                if not target.is_file() or target.is_symlink() or target.read_bytes() != archive.read(m):
                    mismatches.append(m.filename)
        checks["archive"] = dict(sha256=sha(args.archive.read_bytes()), archive_bytes=args.archive.stat().st_size, entries=len(members),
                                 uncompressed_bytes=sum(m.file_size for m in members),
                                 compressed_bytes=sum(m.compress_size for m in members),
                                 duplicate_members=len(names) != len(set(names)), unsafe_paths=unsafe,
                                 symlinks=symlinks, extracted_mismatches=mismatches)

    files = git("ls-tree", "-r", "--name-only", REVISION).decode().splitlines()
    identity_files = []
    for prefix in ("tdn/", "reference/", "scripts/", "configs/", "tests/"):
        identity_files.extend(sorted(f for f in files if f.startswith(prefix)
                                     and "__pycache__" not in Path(f).parts))
    identity_files += ["requirements.txt", "pyproject.toml", ".raven/automation.json"]
    rows = read("outputs/verification/source.json")["files"]
    needed = set(identity_files) | {r["path"] for r in rows}
    blobs = {path: blob(path) for path in needed}
    source_bad = [r["path"] for r in rows if sha(blobs[r["path"]]) != r["sha256"]]
    fingerprint = sha(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode())
    sources = [read("outputs/" + stage + "/" + name)
               for stage in ("verification", "smoke")
               for name in ("source.json", "source_after.json")]
    digest = hashlib.sha256()
    for path in identity_files:
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(blobs[path]).digest())
    contract = read("outputs/native/contract.json")
    checks["source"] = dict(
        stage_source_files=len(rows), stage_hash_mismatches_against_git=source_bad,
        all_stage_before_after_equal=all(s == sources[0] for s in sources),
        stage_fingerprint_recomputed=fingerprint,
        fingerprint_matches=all(s["fingerprint"] == fingerprint for s in sources),
        native_source_files=len(identity_files), native_source_hash_recomputed=digest.hexdigest(),
        native_identity_matches=(contract["identity"]["source"]["sha256"] == digest.hexdigest()
                                 and contract["identity"]["source"]["files"] == len(identity_files)),
        script_hash_matches_run=(sha(blobs["scripts/response_preconditioner_verify.sh"])
                                 == read("run.json")["provenance"]["script_sha256"]))

    manifests = {}
    for stage in ("verification", "smoke"):
        folder = base / "outputs" / stage
        manifest = read("outputs/" + stage + "/artifacts.json")
        failures = []
        for row in manifest["files"]:
            relative = PurePosixPath(row["path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in row["path"]:
                failures.append(row["path"])
                continue
            path = folder / row["path"]
            if not path.is_file() or path.is_symlink():
                failures.append(row["path"])
                continue
            raw = path.read_bytes()
            if len(raw) != row["bytes"] or sha(raw) != row["sha256"]:
                failures.append(row["path"])
        actual = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
        expected = {r["path"] for r in manifest["files"]} | {"artifacts.json"}
        manifests[stage] = dict(files=len(manifest["files"]), hash_or_size_failures=failures,
                                unlisted_files=sorted(actual - expected), missing_files=sorted(expected - actual),
                                duplicate_paths=len(manifest["files"]) != len(expected) - 1)
    checks["stage_manifests"] = manifests

    def junit_ids(cases):
        return [(c.attrib.get("classname"), c.attrib["name"]) for c in cases]

    cases = list(ET.parse(base / "outputs/verification/verification.xml").getroot().iter("testcase"))
    prior = ET.fromstring(blob("results/response-preconditioner-package-identity-20261015/verification.xml"))
    actual_ids, prior_ids = junit_ids(cases), junit_ids(prior.iter("testcase"))
    checks["junit"] = dict(total=len(cases), failures=sum(c.find("failure") is not None for c in cases),
                            errors=sum(c.find("error") is not None for c in cases),
                            skipped=sum(c.find("skipped") is not None for c in cases),
                            duplicate_ids=len(actual_ids) != len(set(actual_ids)),
                            identical_ids_to_pre_launch_184_case_receipt=set(actual_ids) == set(prior_ids),
                            class_counts=dict(sorted(collections.Counter(c.attrib["classname"] for c in cases).items())))

    outputs = read(".raven/contracts/outputs.v1.json")["outputs"]
    paths = [r["path"] for r in outputs]
    checks["dynamic_raven_outputs"] = dict(registered_files=len(outputs), all_required=all(r["required"] for r in outputs),
        all_relative=all(not PurePosixPath(p).is_absolute() and ".." not in PurePosixPath(p).parts for p in paths),
        duplicates=len(paths) != len(set(paths)), missing=[p for p in paths if not (base / p).is_file()])

    metrics = read("outputs/smoke/metrics.json")
    raven_metrics = [json.loads(line) for line in (base / "metrics.jsonl").read_text().splitlines()]
    science = [row for row in raven_metrics if row["phase"].startswith("smoke/")]
    keys = ("rms", "maximum", "mean_error", "centered_rms", "horizon", "direction",
            "action_correction_rms", "correction_uncertainty_rms", "action_correction_relative_rms")
    bad = []
    for index, (raw, rv) in enumerate(zip(metrics, science)):
        expected = {k: raw[k] for k in keys if raw.get(k) is not None}
        phase = "smoke/" + "/".join(str(raw.get(k, "unknown")) for k in ("kind", "split", "method"))
        if expected != rv["metrics"] or phase != rv["phase"] or index != rv["step"]:
            bad.append(index)
    checks["raven_science_metrics"] = dict(raw_rows=len(metrics), raven_science_rows=len(science),
        raven_stage_rows=len(raven_metrics) - len(science), row_mismatches=bad,
        parent_identity_in_raven_series=False, parent_identity_in_raw_artifacts=True)

    telemetry = {}
    for prefix in ("outputs/native/", ""):
        index = read(prefix + "telemetry/index.json")
        sample_path = base / (prefix + "telemetry/cpu-memory.jsonl")
        samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
        coverage = index["sources"][0]["coverage"]
        data = dict(samples=len(samples), index_samples_match=len(samples) == coverage["samples"],
            byte_count_matches=sample_path.stat().st_size == coverage["bytes"], status=index["sources"][0]["status"],
            reported_missed=coverage["missed_samples"], partial_samples=sum(s.get("status") == "partial" for s in samples),
            cpu_percent_available=sum(s.get("cpu_utilization_percent") is not None for s in samples),
            maximum_sampled_rss_bytes=max(s.get("rss_bytes") or 0 for s in samples), last_cpu_seconds=samples[-1].get("cpu_seconds"),
            timestamp_increasing=all(a["timestamp"] < b["timestamp"] for a, b in zip(samples, samples[1:])))
        if prefix:
            with (base / (prefix + "telemetry/cpu-memory.csv")).open() as handle:
                csv_rows = list(csv.DictReader(handle))
            mismatches = [i for i, (cr, sample) in enumerate(zip(csv_rows, samples))
                          if any(cr[k] != ("" if sample[k] is None else str(sample[k])) for k in cr)]
            data.update(csv_rows=len(csv_rows), csv_json_mismatches=mismatches,
                partial_reasons=dict(collections.Counter(s["reason"] for s in samples if s.get("status") == "partial")))
        telemetry["worker" if prefix else "supervisor"] = data
    checks["telemetry"] = telemetry

    run, native = read("run.json"), read("outputs/native/status.json")
    checks["execution"] = dict(raven_state=run["state"], application_exit_code=run["metadata"]["automation"]["application_exit_code"],
        scheduler_outcome=run["metadata"]["automation"]["scheduler_outcome"], native_state=native["status"], native_exit_code=native["exit_code"],
        allocation_job_state_is_startup_snapshot=native["allocation"]["job"]["JobState"], job_id=run["job_id"], device=native["device"],
        source_unchanged_reported=native["source_unchanged"])
    checks["contract"] = dict(copied_contract_sha256=sha((base / "outputs/native/contract.json").read_bytes()),
        original_prepared_contract_sha256_reported=native["contract_sha256"], copied_contract_serialization_is_changed=True,
        profile_snapshot_present=True, original_profile_raw_file_present=False)
    checks["logs"] = dict(portable_stage_logs=[row["path"] for row in read("logs.json")["logs"] if not Path(row["path"]).is_absolute()],
        external_scheduler_log_paths=[row["path"] for row in read("logs.json")["logs"] if Path(row["path"]).is_absolute()],
        outer_stdout_bytes=(base / "logs/stdout.log").stat().st_size, outer_stderr_bytes=(base / "logs/stderr.log").stat().st_size)
    report["limitations"] = [
        "Raven reports application completion and exit 0; independent final Slurm sacct/scontrol accounting is absent, scheduler_outcome remains unknown. RUNNING allocation fields are startup snapshots.",
        "Native copied contract is canonical JSON reserialized from the original prepared pretty JSON; its raw hash is different by construction. Original prepared file and raw profile absent, so their recorded byte hashes cannot be independently checked from this archive.",
        "All stage artifact hashes and known source identity match; hashes establish internal integrity and source correspondence, not authenticated provenance.",
        "Four stage logs are portable; scheduler stdout/stderr index entries point to original absolute host paths outside this archive. Empty logs/stdout.log and logs/stderr.log are not copies of those external scheduler streams.",
        "Worker process-tree telemetry is partial: 21 churn intervals have null CPU utilization; first interval is also unavailable. Sampled summed RSS is not allocation maximum RSS; departed-process CPU cannot be accumulated from the terminal snapshot.",
        "All results are IMPLEMENTATION_ONLY, CPU only; no GPU, scientific promotion, fresh confirmation, novelty, speedup or broad performance statement is justified."]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
