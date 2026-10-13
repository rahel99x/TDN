#!/usr/bin/env python3
"""Tangent campaign adapter for the shared strict Raven launch context."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    import response_diagnostic_raven as shared
except ModuleNotFoundError:
    from scripts import response_diagnostic_raven as shared

SCRIPT = "scripts/tangent_response_campaign.sh"
SCHEMA = "tdn.tangent-response-campaign/v1"


def context(snapshot=None):
    return shared.context(snapshot, script=SCRIPT)


association = shared.association


def receipt_data(ctx, path):
    path = shared.confined(path, ctx["root"])
    value, checksum = shared.read_json(path, root=ctx["root"], limit=shared.DECISION_BYTES)
    if (value.get("schema") != SCHEMA or value.get("raven_launch") != association(ctx)
            or value.get("project_root", str(ctx["root"])) != str(ctx["root"])):
        raise ValueError("Tangent campaign receipt does not belong to this exact local Raven launch")
    run_id = shared.identity(value.get("run_id", value.get("parent_run_id")), "tangent campaign")
    if not run_id.startswith("tangent-response-"):raise ValueError("Tangent campaign needs its separate program identity")
    ids = value.get("run_ids")
    if not isinstance(ids, dict):raise ValueError("Tangent child run IDs must be an explicit phase mapping")
    for phase, child in ids.items():
        shared.identity(phase, "tangent phase")
        if not shared.identity(child, "tangent child").startswith("tangent-response-"):
            raise ValueError("Tangent child run belongs to a different program")
    if len(set(ids.values())) != len(ids):raise ValueError("Tangent phases require independent child run IDs")
    if value.get("status") not in {"RUNNING", "COMPLETED", "FAILED", "INTERRUPTED"}:
        raise ValueError("Invalid tangent coordinator infrastructure state")
    if not isinstance(value.get("phase"), str) or not value["phase"].isprintable() or len(value["phase"]) > 128:
        raise ValueError("Invalid tangent coordinator phase")
    if not isinstance(value.get("events"), list):raise ValueError("Tangent campaign events must be retained")
    return path, value, checksum


def publish(ctx, receipt, *, status_file=None):
    path, value, checksum = receipt_data(ctx, receipt)
    output = ctx["run_dir"]/"outputs"
    paths = dict(schema="tdn.tangent-response-research-paths/v1", path_base="TDN project root",
        campaign_receipt=dict(path=path.relative_to(ctx["root"]).as_posix(), sha256=checksum), runs={},
        scope="Coordinator infrastructure counts; literal scientific decisions and measurements live in child research projects")
    for phase, child in value["run_ids"].items():
        base = "runs/"+child
        pointers = dict(run_id=child, raven_project=base+"/raven", project_catalog=base+"/raven/project_catalog.json",
            portable_research_bundle=base+"/report/tangent-response-raven.zip",
            portable_research_manifest=base+"/report/tangent-response-raven.manifest.json")
        for key, pointer in pointers.items():
            if key != "run_id":shared.confined(ctx["root"]/shared.relative(pointer), ctx["root"])
        paths["runs"][phase] = pointers
    expected = None
    previous = output/"research_paths.json"
    if status_file is None and previous.exists():
        retained, _ = shared.read_json(previous, root=ctx["run_dir"])
        latest = retained.get("latest_native_status")
        if latest is not None:
            source = latest.get("source", {})
            expected = source.get("sha256")
            if not isinstance(expected, str) or not shared.SHA256.fullmatch(expected):
                raise ValueError("Retained tangent status requires its source SHA256")
            status_file = shared.confined(ctx["root"]/shared.relative(source.get("path")), ctx["root"])
    metrics = dict(campaign_events_recorded=len(value["events"]), campaign_completed=int(value["status"] == "COMPLETED"),
                   campaign_failed=int(value["status"] == "FAILED"), declared_child_runs=len(value["run_ids"]))
    progress = []
    if status_file is not None:
        source, _ = shared.read_json(status_file, root=ctx["root"], limit=shared.DECISION_BYTES)
        run_id = source.get("run_id")
        if run_id not in value["run_ids"].values():raise ValueError("Native status belongs to another tangent campaign")
        counts, status = shared.status_counts(ctx, dict(smoke_run_id=run_id, diagnostic_run_id=run_id), status_file, expected=expected)
        paths["latest_native_status"] = status;metrics.update(counts)
        if counts["registered_units"]:
            progress = ["--completed", str(counts["terminal_jobs"]), "--total", str(counts["registered_units"]),
                        "--unit", "registered native allocations"]
    gates = value.get("gates", {})
    if not isinstance(gates, dict):raise ValueError("Tangent gates must retain their exact phase mapping")
    decisions = dict(schema="tdn.tangent-response-campaign-decisions/v1", gates=gates,
        source=dict(path=path.relative_to(ctx["root"]).as_posix(), sha256=checksum),
        scope="Literal scientific decisions; coordinator completion does not promote a negative or unresolved gate")
    shared.write_json(output/"campaign.json", value, root=ctx["run_dir"])
    shared.write_json(output/"research_paths.json", paths, root=ctx["run_dir"])
    shared.write_json(output/"decisions.json", decisions, root=ctx["run_dir"])
    shared.helper(ctx, "metric", "--phase", "tangent_campaign/"+value["phase"], "--step", str(len(value["events"])),
        *[argument for key, number in metrics.items() for argument in ("--value", key+"="+str(number))], *progress)


def initialize(ctx, receipt):
    receipt_data(ctx, receipt)
    for suffix in ("stdout", "stderr"):shared.register_log(ctx, suffix)
    publish(ctx, receipt)
    for name in ("campaign.json", "research_paths.json", "decisions.json"):
        shared.register_artifact(ctx, "outputs/"+name)


def main(argv=None):
    if not os.environ.get("RAVEN_LAUNCH_PLAN"):return 0
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check");check.add_argument("--snapshot", type=Path, required=True)
    init = commands.add_parser("initialize");init.add_argument("--snapshot", type=Path, required=True);init.add_argument("--receipt", type=Path, required=True)
    for name in ("publish", "progress", "finish"):
        command = commands.add_parser(name);command.add_argument("--receipt", type=Path, required=True);command.add_argument("--status-file", type=Path)
    args = parser.parse_args(argv)
    try:
        ctx = context(getattr(args, "snapshot", None))
        if args.command == "initialize":initialize(ctx, args.receipt)
        elif args.command != "check":publish(ctx, args.receipt, status_file=args.status_file)
        return 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError, subprocess.CalledProcessError) as error:
        print("TDN tangent Raven: "+str(error), file=sys.stderr);return 2


if __name__ == "__main__":raise SystemExit(main())
