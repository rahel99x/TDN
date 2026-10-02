#!/usr/bin/env python3
"""Read-only live Slurm policy checks; never allocate or modify cluster state."""
from __future__ import annotations

import argparse
import getpass
import json
import re
import shutil
import subprocess
from pathlib import Path

ACCOUNT = "anakano_81"
USER = "aadaniel"


def command(*args: str) -> str:
    if not shutil.which(args[0]):
        raise RuntimeError(f"Required read-only discovery command unavailable: {args[0]}")
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr.strip()}")
    return result.stdout


def association_verified(text: str, partition: str) -> bool:
    for line in text.splitlines():
        cells = line.strip().split("|")
        if len(cells) >= 3 and cells[0] == ACCOUNT and cells[1] == USER:
            if cells[2] in ("", partition):
                return True
    return False


def myaccount_verified(text: str) -> bool:
    # myaccount reports associations of the calling user. Match complete tokens;
    # account suffixes, prefixes and a prose error message must never authorize.
    for line in text.splitlines():
        if re.search(r"(?i)\b(error|denied|unauthorized|not authorized|invalid)\b", line):
            continue
        tokens = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", line)
        if tokens and tokens[0] == ACCOUNT:
            return True
    return False


def seconds(value: str) -> int:
    if value.upper() in ("UNLIMITED", "INFINITE"):
        return 2**62
    if not re.fullmatch(r"(?:\d+-)?\d+:\d{2}(?::\d{2})?", value):
        raise RuntimeError(f"Ambiguous Slurm walltime: {value}")
    days, _, clock = value.rpartition("-")
    parts = [int(x) for x in clock.split(":")]
    if len(parts) == 2:
        parts = [0, *parts]
    return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def partition_fields(raw: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=([^\s]+)", raw))


def account_allowed(fields: dict[str, str]) -> bool:
    allowed = fields.get("AllowAccounts")
    denied = fields.get("DenyAccounts", "").split(",")
    if allowed is None:
        raise RuntimeError("Partition account policy unavailable")
    return (allowed == "ALL" or ACCOUNT in allowed.split(",")) and ACCOUNT not in denied


def check(partition: str, walltime: str, gpu: bool, cpus: int = 1, mem_gib: int = 1) -> dict:
    if getpass.getuser() != USER:
        raise RuntimeError(f"Expected current CARC user {USER}")
    raw = command("scontrol", "show", "partition", partition)
    fields = partition_fields(raw)
    if fields.get("PartitionName") != partition or fields.get("State") != "UP":
        raise RuntimeError(f"Partition {partition} is missing or not UP")
    if not account_allowed(fields):
        raise RuntimeError(f"Partition {partition} does not authorize {ACCOUNT}")
    if seconds(walltime) > seconds(fields.get("MaxTime", "UNKNOWN")):
        raise RuntimeError("Requested walltime exceeds the live partition limit")
    associations = ""
    verified_by = ""
    try:
        associations = command("sacctmgr", "-nP", "show", "associations", "where",
                               f"user={USER}", f"account={ACCOUNT}",
                               "format=Account,User,Partition,QOS,MaxWall,GrpTRES")
        if association_verified(associations, partition):
            verified_by = "sacctmgr exact account/user/partition association"
    except RuntimeError:
        pass
    account_report = ""
    if not verified_by:
        account_report = command("myaccount")
        if not myaccount_verified(account_report):
            raise RuntimeError(f"Cannot verify current user's exact account {ACCOUNT}")
        verified_by = "myaccount exact account token for current user"
    nodes = command("sinfo", "-p", partition, "-N", "-h", "-o", "%N|%G|%f|%m|%c")
    eligible_nodes = []
    for line in nodes.splitlines():
        cells = line.strip().split("|")
        if len(cells) != 5:
            continue
        try:
            if int(cells[3]) >= mem_gib * 1024 and int(cells[4]) >= cpus:
                eligible_nodes.append(cells)
        except ValueError:
            continue
    if not eligible_nodes:
        raise RuntimeError("No node advertises the requested host memory and CPU count")
    if gpu:
        good_nodes = []
        for cells in eligible_nodes:
            features = cells[2].split(",")
            if re.search(r"(?:^|[, ])gpu:a100:\d+", cells[1]) and "a100-40gb" in features:
                good_nodes.append(cells[0])
        if not good_nodes:
            raise RuntimeError("Live GPU partition has no advertised a100 + a100-40gb node")
    else:
        if any("gpu:" in line.split("|")[1] for line in nodes.splitlines() if "|" in line):
            raise RuntimeError("Selected CPU partition advertises GPUs; choose a CPU-only partition")
    qos_report = None
    if shutil.which("sacctmgr"):
        try:
            qos_report = command("sacctmgr", "-nP", "show", "qos",
                                 "format=Name,MaxWall,MaxTRESPerJob,MaxJobsPU")
        except RuntimeError:
            pass
    max_cpus = fields.get("MaxCPUsPerNode", "UNLIMITED")
    if max_cpus.isdecimal() and cpus > int(max_cpus):
        raise RuntimeError("Requested CPUs exceed live partition limit")
    max_mem = fields.get("MaxMemPerNode", "UNLIMITED")
    if max_mem.isdecimal() and mem_gib * 1024 > int(max_mem):
        raise RuntimeError("Requested memory exceeds live partition limit")
    return {"category": "newly_measured_result", "account": ACCOUNT, "user": USER,
            "partition": partition, "requested_walltime": walltime, "gpu": gpu,
            "requested_cpus": cpus, "requested_host_mem_gib": mem_gib,
            "authorization_evidence": verified_by, "partition_report": raw,
            "association_report": associations, "myaccount_report": account_report,
            "node_report": nodes, "qos_report": qos_report,
            "note": "Read-only discovery; allocation admission still belongs to Slurm."}


def discover_cpu(walltime: str, cpus: int = 1, mem_gib: int = 1) -> str:
    inventory = command("sinfo", "-h", "-o", "%P|%a|%G|%l|%c|%m")
    candidates = set()
    for line in inventory.splitlines():
        cells = line.strip().split("|")
        if len(cells) == 6 and cells[1] == "up" and "gpu:" not in cells[2]:
            candidates.add(cells[0].rstrip("*"))
    for partition in sorted(candidates):
        try:
            check(partition, walltime, False, cpus, mem_gib)
            return partition
        except RuntimeError:
            continue
    raise RuntimeError("No live authorized CPU-only partition found; inspect probe_carc.sh")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition")
    parser.add_argument("--walltime", required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--mem-gib", type=int, default=1)
    parser.add_argument("--discover-cpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.discover_cpu:
            print(discover_cpu(args.walltime, args.cpus, args.mem_gib))
            return
        if not args.partition:
            parser.error("--partition is required for verification")
        report = check(args.partition, args.walltime, args.gpu, args.cpus, args.mem_gib)
        if args.output:
            root = Path(__file__).resolve().parents[1]
            output = args.output.resolve()
            if not output.is_relative_to(root):
                raise RuntimeError("Policy report leaves the project")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    except RuntimeError as exc:
        parser.exit(2, f"TDN live policy: {exc}\n")


if __name__ == "__main__":
    main()
