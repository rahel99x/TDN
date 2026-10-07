#!/usr/bin/env python3
"""Run one provenance-bound stage of the M00–M23 bounded roadmap."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STAGES = ("audit", "headroom", "prepare", "train", "confirm_prepare", "confirm", "policy", "transfer", "scaling", "report")
GPU_STAGES = ("train", "confirm", "policy", "scaling")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def parse_prerequisites(values, stage):
    result = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not path or label in result or label not in STAGES:
            raise ValueError("Prerequisites must be unique STAGE=PATH declarations")
        result[label] = Path(path)
    if set(result) != set(STAGES[:STAGES.index(stage)]):
        raise ValueError("Every preceding roadmap stage is required exactly once")
    return result


def compatible_software(value):
    keys = ("python", "executable", "venv", "torch", "numpy", "scipy", "torch_cuda_runtime",
            "git_commit", "source_tree_sha256")
    return {key: value.get(key) for key in keys}


def verify_execution(path, protocol, *, source_tree_sha256=None):
    """Bind root-engine science artifacts to immutable worker provenance."""
    from tdn.analysis.roadmap.engine import verify_science
    path = Path(path)
    science = verify_science(protocol, path)
    seal = json.loads((path / "workflow-seal.json").read_text())
    if seal.get("schema_version") != 1 or seal.get("protocol_sha256") != digest(protocol):
        raise ValueError("Roadmap execution seal protocol differs")
    expected = ("execution.json", "protocol.json", "stage.json", "science_manifest.json")
    if set(seal.get("files", {})) != set(expected):
        raise ValueError("Roadmap execution seal has an incomplete file inventory")
    for name in expected:
        file = path / name
        if file.is_symlink() or not file.is_file() or file_digest(file) != seal["files"][name]:
            raise ValueError(f"Roadmap execution evidence changed: {name}")
    execution = json.loads((path / "execution.json").read_text())
    record = json.loads((path / "stage.json").read_text())
    if record.get("status") != "COMPLETED" or execution.get("protocol_sha256") != digest(protocol):
        raise ValueError("Roadmap stage did not complete under its declared protocol")
    if source_tree_sha256 is not None and execution.get("software", {}).get("source_tree_sha256") != source_tree_sha256:
        raise ValueError("Roadmap stage execution source differs")
    return {"science": science, "execution": execution, "seal": seal}


def seal_execution(path, protocol):
    from tdn.runtime.metadata import write_json
    path = Path(path)
    write_json(path / "workflow-seal.json", {"schema_version": 1, "protocol_sha256": digest(protocol),
        "files": {name: file_digest(path / name) for name in
                  ("execution.json", "protocol.json", "stage.json", "science_manifest.json")}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--profile", choices=("smoke", "development", "full"), default="full")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prerequisite-dir", action="append", default=[], metavar="STAGE=PATH")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--local-root", type=Path)
    args = parser.parse_args(argv)
    run_dir = report = record = None
    owned, exit_code = False, 1
    previous_tower = os.environ.get("TDN_TOWER_DIR")
    started = time.monotonic()
    try:
        from tdn.runtime.storage import CARC_ROOT, configure_storage, contained_path
        from tdn.runtime.preflight import execution_mode, verify_runtime
        if args.local_root is not None:
            if args.local_root.resolve() != ROOT or ROOT == CARC_ROOT or any(
                    os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_STEP_ID", "TDN_EXECUTION_MODE")):
                raise ValueError("Local roadmap requires this CPU checkout without scheduler or desktop overrides")
            if args.device != "cpu" or args.profile == "full":
                raise ValueError("Local roadmap supports CPU smoke/development; full confirmation requires Fedora Slurm")
        mode = execution_mode()
        if mode not in ("local-cpu", "desktop-slurm"):
            raise ValueError("Roadmap supports Fedora Slurm and explicit local CPU development")
        if mode == "local-cpu" and (args.local_root is None or args.profile == "full" or args.device != "cpu"):
            raise ValueError("Local CPU work requires --local-root and a disjoint smoke/development profile")
        if mode == "desktop-slurm" and args.device != ("cuda" if args.stage in GPU_STAGES else "cpu"):
            raise ValueError("Allocated roadmap stages must use their frozen CPU/GPU device")
        configure_storage()
        verify_runtime(args.device, "roadmap-" + args.stage)
        from tdn.analysis.roadmap.protocol import build_protocol, validate_protocol
        from tdn.analysis.roadmap.engine import run_stage, verify_science
        from tdn.runtime.metadata import software_metadata, write_json
        from tdn.runtime.precision import reference_precision
        from tdn.runtime.signal_handling import StopRequest
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        reference_precision()
        software = software_metadata()
        protocol = build_protocol(args.profile)
        validate_protocol(protocol)
        site = None
        if mode == "desktop-slurm":
            from tdn.runtime.desktop_slurm import load_profile
            site = load_profile()
        paths = {key: contained_path(value) for key, value in parse_prerequisites(args.prerequisite_dir, args.stage).items()}
        candidate = contained_path(args.run_dir)
        if candidate.exists():
            raise ValueError("Preserve prior results; choose a fresh nonexistent stage directory")
        lineage = {}
        for label in STAGES[:STAGES.index(args.stage)]:
            path = paths[label]
            if candidate == path or candidate.is_relative_to(path) or path.is_relative_to(candidate):
                raise ValueError("Stage and prerequisite science directories must be separate")
            try:
                verified = verify_execution(path, protocol, source_tree_sha256=software["source_tree_sha256"])
                execution = verified["execution"]
                if execution.get("stage") != label or execution.get("execution_mode") != mode:
                    raise ValueError("Prerequisite stage or execution mode differs")
                if compatible_software(execution.get("software", {})) != compatible_software(software):
                    raise ValueError("Prerequisite software differs from this execution environment")
                if site is not None and execution.get("slurm_profile_sha256") != digest(site):
                    raise ValueError("Prerequisite belongs to a different Fedora Slurm profile")
                expected_prior = {key: lineage[key] for key in STAGES[:STAGES.index(label)]}
                if execution.get("prerequisites", {}) != expected_prior:
                    raise ValueError("Prerequisites belong to different roadmap lineages")
                lineage[label] = {"run_dir": str(path), "workflow_seal_sha256": file_digest(path / "workflow-seal.json")}
            except (ValueError, OSError, KeyError, TypeError) as error:
                if args.stage != "report":
                    raise
                lineage[label] = {"run_dir": str(path), "verification": "INVALID" if path.exists() else "MISSING",
                                  "error": f"{type(error).__name__}: {error}"}
        candidate.mkdir(parents=True, exist_ok=False)
        run_dir = candidate
        execution = {"stage": args.stage, "profile": args.profile, "device": args.device, "execution_mode": mode,
            "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
            "software": software, "protocol_sha256": digest(protocol), "prerequisites": lineage}
        if os.environ.get("TDN_ROADMAP_PROTOCOL_SHA256"):
            execution["workflow_protocol_sha256"] = os.environ["TDN_ROADMAP_PROTOCOL_SHA256"]
        if site is not None:
            execution["slurm_profile_sha256"] = digest(site)
        write_json(run_dir / "execution.json", execution)
        write_json(run_dir / "protocol.json", protocol)
        record = {**execution, "status": "RUNNING", "actually_ran": False, "benchmark_suite": "roadmap",
                  "training_attempted": args.stage == "train"}
        write_json(run_dir / "stage.json", record)
        from tdn.reporting import attach_report, emit
        report, owned = attach_report(run_dir, name="TDN/roadmap/" + args.stage, script="scripts/roadmap.py",
            parameters={"device": args.device, "stage": args.stage, "benchmark_suite": "roadmap", "profile": args.profile,
                        "execution_mode": mode, "protocol_sha256": digest(protocol),
                        "source_tree_sha256": software["source_tree_sha256"]})
        os.environ["TDN_TOWER_DIR"] = str(report)
        record["actually_ran"] = True
        write_json(run_dir / "stage.json", record)
        emit({"stage_started": 1}, phase="roadmap/" + args.stage)
        with StopRequest() as stop:
            result = run_stage(protocol, args.stage, run_dir, prerequisites=paths, device=args.device, stop=stop)
            if stop.requested:
                raise InterruptedError("Stop requested before sealing roadmap evidence")
        if result.get("status") != "COMPLETED":
            if result.get("status") in ("INCOMPLETE", "INTERRUPTED"):
                raise InterruptedError("Roadmap stage stopped with partial evidence")
            raise RuntimeError("Roadmap stage did not complete; partial evidence remains unsealed")
        if software_metadata()["source_tree_sha256"] != software["source_tree_sha256"]:
            raise RuntimeError("Execution source changed during the stage; choose a fresh run")
        if digest(json.loads((run_dir / "protocol.json").read_text())) != digest(protocol):
            raise RuntimeError("Protocol changed during stage execution")
        if site is not None and digest(load_profile()) != digest(site):
            raise RuntimeError("Fedora Slurm profile changed during stage execution")
        if args.stage != "report":
            for label, path in paths.items():
                verify_execution(path, protocol, source_tree_sha256=software["source_tree_sha256"])
                if file_digest(path / "workflow-seal.json") != lineage[label]["workflow_seal_sha256"]:
                    raise RuntimeError("A prerequisite changed during execution")
        verify_science(protocol, run_dir)
        record.update(status="COMPLETED", elapsed_seconds=time.monotonic() - started)
        write_json(run_dir / "stage.json", record)
        seal_execution(run_dir, protocol)
        emit({"stage_completed": 1}, phase="roadmap/" + args.stage)
        if (run_dir / "summary.txt").is_file():
            print((run_dir / "summary.txt").read_text(), end="")
        print(json.dumps({"status": "COMPLETED", "stage": args.stage, "summary": str(run_dir / "summary.json"),
                          "scientific_outcome": result.get("scientific_outcome"), "tower": str(report)}, indent=2))
        exit_code = 0
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, (InterruptedError, TimeoutError, KeyboardInterrupt))
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 75 if interrupted else 1
        if record is not None:
            record.update(status="INTERRUPTED" if interrupted else "FAILED", error=f"{type(error).__name__}: {error}",
                          elapsed_seconds=time.monotonic() - started)
            write_json(run_dir / "stage.json", record)
        print(f"TDN roadmap: {type(error).__name__}: {error}", file=sys.stderr)
    finally:
        try:
            if owned:
                from tdn.cli import finalize_tower_report
                if not finalize_tower_report(report, source_dirs=[run_dir], state=record["status"], started=started,
                    exit_code=exit_code, metadata={"device": args.device, "actually_ran": record["actually_ran"],
                        "stage": args.stage, "scientific_status": record["status"], "benchmark_suite": "roadmap"}):
                    exit_code = exit_code or 1
        finally:
            if previous_tower is None:
                os.environ.pop("TDN_TOWER_DIR", None)
            else:
                os.environ["TDN_TOWER_DIR"] = previous_tower
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
