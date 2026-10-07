#!/usr/bin/env python3
"""Execute one sealed stage of the bounded research agenda in the project venv."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_prerequisites(values, stage):
    from tdn.analysis.agenda.protocol import STAGES
    result = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not path or label in result or label not in STAGES:
            raise ValueError("Prerequisites must be unique STAGE=PATH declarations")
        result[label] = Path(path)
    if set(result) != set(STAGES[:STAGES.index(stage)]):
        raise ValueError("Every preceding agenda stage is required exactly once")
    return result


def compatible_software(value):
    """Compare the actual environment while permitting distinct scheduler jobs."""
    keys = ("python", "executable", "venv", "torch", "numpy", "scipy", "torch_cuda_runtime",
            "git_commit", "source_tree_sha256")
    return {key: value.get(key) for key in keys}


def main(argv=None):
    from tdn.analysis.agenda.protocol import STAGES, PROFILES, GPU_STAGES, build_protocol
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
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
                raise ValueError("Local agenda requires this CPU checkout without scheduler or desktop overrides")
            if args.device != "cpu" or args.profile == "full":
                raise ValueError("Local agenda supports CPU smoke/development; full confirmation requires Fedora Slurm")
        mode = execution_mode()
        if mode not in ("local-cpu", "desktop-slurm"):
            raise ValueError("Agenda supports Fedora Slurm and explicit local CPU development")
        if mode == "local-cpu" and (args.local_root is None or args.profile == "full" or args.device != "cpu"):
            raise ValueError("Local CPU work requires --local-root and a disjoint smoke/development profile")
        if args.stage not in GPU_STAGES and (args.device != "cpu" or args.dataset_dir is not None):
            raise ValueError("Structure and preparation are CPU stages without a prepared dataset input")
        if args.stage in GPU_STAGES and args.dataset_dir is None:
            raise ValueError("Learned/confirmation/policy stages require the sealed prepared dataset")
        if mode == "desktop-slurm" and args.device != ("cuda" if args.stage in GPU_STAGES else "cpu"):
            raise ValueError("Allocated agenda stages must use their frozen CPU/GPU device")
        configure_storage()
        verify_runtime(args.device, "agenda-" + args.stage)
        from tdn.analysis.agenda.artifacts import digest, file_digest, seal_stage, verify_stage
        from tdn.runtime.metadata import software_metadata, write_json
        from tdn.runtime.precision import reference_precision
        from tdn.runtime.signal_handling import StopRequest
        import torch
        torch.set_num_threads(1)
        # Each allocated stage is a fresh process; imported interactive tools may
        # have already initialized the inter-op pool, so retain its valid state.
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        reference_precision()
        software = software_metadata()
        protocol = build_protocol(args.profile)
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
            manifest = verify_stage(path, stage=label, profile=args.profile,
                                    source_tree_sha256=software["source_tree_sha256"])
            execution = json.loads((path / "execution.json").read_text())
            if manifest["protocol_sha256"] != digest(protocol) or execution.get("execution_mode") != mode:
                raise ValueError("Prerequisite protocol or execution mode differs")
            if compatible_software(execution.get("software", {})) != compatible_software(software):
                raise ValueError("Prerequisite software differs from this execution environment")
            if site is not None and execution.get("slurm_profile_sha256") != digest(site):
                raise ValueError("Prerequisite belongs to a different Fedora Slurm profile")
            # Every prerequisite must describe this same chain, not a mixture
            # of individually valid stages from different workflows.
            expected_prior = {key: lineage[key] for key in STAGES[:STAGES.index(label)]}
            if execution.get("prerequisites", {}) != expected_prior:
                raise ValueError("Prerequisites belong to different agenda lineages")
            lineage[label] = {"run_dir": str(path), "manifest_sha256": file_digest(path / "manifest.json")}
        dataset = contained_path(args.dataset_dir) if args.dataset_dir is not None else None
        if dataset is not None and dataset != paths.get("prepare"):
            raise ValueError("Dataset must be the prepare stage from this exact prerequisite chain")
        candidate.mkdir(parents=True, exist_ok=False)
        run_dir = candidate
        command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
        execution = {"stage": args.stage, "profile": args.profile, "device": args.device,
            "execution_mode": mode, "command": command, "software": software,
            "protocol_sha256": digest(protocol), "prerequisites": lineage}
        if os.environ.get("TDN_AGENDA_PROTOCOL_SHA256"):
            execution["workflow_protocol_sha256"] = os.environ["TDN_AGENDA_PROTOCOL_SHA256"]
        if site is not None:
            execution["slurm_profile_sha256"] = digest(site)
        if dataset is not None:
            execution["dataset_dir"] = str(dataset)
            execution["dataset_manifest_sha256"] = file_digest(dataset / "manifest.json")
        write_json(run_dir / "execution.json", execution)
        write_json(run_dir / "protocol.json", protocol)
        record = {**execution, "status": "RUNNING", "actually_ran": False,
                  "benchmark_suite": "agenda", "training_attempted": args.stage in ("controls", "optimize", "compression", "kernel")}
        write_json(run_dir / "stage.json", record)
        from tdn.reporting import attach_report, emit
        report, owned = attach_report(run_dir, name="TDN/agenda/" + args.stage,
            script="scripts/agenda.py", parameters={"device": args.device, "stage": args.stage,
                "benchmark_suite": "agenda", "profile": args.profile, "execution_mode": mode,
                "protocol_sha256": digest(protocol), "source_tree_sha256": software["source_tree_sha256"]})
        os.environ["TDN_TOWER_DIR"] = str(report)
        record["actually_ran"] = True
        write_json(run_dir / "stage.json", record)
        emit({"stage_started": 1}, phase="agenda/" + args.stage)
        with StopRequest() as stop:
            if args.stage == "structure":
                from tdn.analysis.agenda.structural import run
                result = run(protocol, run_dir, stop=stop)
            else:
                from tdn.analysis.agenda import engine
                if args.stage == "prepare":
                    result = engine.prepare(protocol, run_dir, stop=stop)
                else:
                    result = engine.run_stage(protocol, args.stage, dataset, paths, run_dir, device=args.device, stop=stop)
            if stop.requested:
                raise InterruptedError("Stop requested before sealing agenda evidence")
        if result.get("status") != "COMPLETED":
            if result.get("status") in ("INCOMPLETE", "INTERRUPTED"):
                raise InterruptedError("Agenda stage stopped with partial evidence")
            raise RuntimeError("Agenda stage did not complete; partial evidence remains unsealed")
        if software_metadata()["source_tree_sha256"] != software["source_tree_sha256"]:
            raise RuntimeError("Execution source changed during the stage; choose a fresh run")
        if digest(json.loads((run_dir / "protocol.json").read_text())) != digest(protocol):
            raise RuntimeError("Protocol changed during stage execution")
        if site is not None and digest(load_profile()) != digest(site):
            raise RuntimeError("Fedora Slurm profile changed during stage execution")
        for label, path in paths.items():
            verify_stage(path, stage=label, profile=args.profile, source_tree_sha256=software["source_tree_sha256"])
            if file_digest(path / "manifest.json") != lineage[label]["manifest_sha256"]:
                raise RuntimeError("A prerequisite changed during execution")
        seal_stage(run_dir, stage=args.stage, profile=args.profile, source_tree_sha256=software["source_tree_sha256"])
        record.update(status="COMPLETED", elapsed_seconds=time.monotonic() - started)
        write_json(run_dir / "stage.json", record)
        emit({"stage_completed": 1}, phase="agenda/" + args.stage)
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
            if not (run_dir / "COMPLETED").exists():
                summary_path = run_dir / "summary.json"
                summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
                summary.update(status=record["status"], scientific_outcome="INCONCLUSIVE", sealing_error=record["error"],
                               stage=args.stage, device=args.device)
                write_json(summary_path, summary)
        print(f"TDN agenda: {type(error).__name__}: {error}", file=sys.stderr)
    finally:
        try:
            if owned:
                from tdn.cli import finalize_tower_report
                if not finalize_tower_report(report, source_dirs=[run_dir], state=record["status"], started=started,
                    exit_code=exit_code, metadata={"device": args.device, "actually_ran": record["actually_ran"],
                        "stage": args.stage, "scientific_status": record["status"], "benchmark_suite": "agenda"}):
                    exit_code = exit_code or 1
        finally:
            if previous_tower is None:
                os.environ.pop("TDN_TOWER_DIR", None)
            else:
                os.environ["TDN_TOWER_DIR"] = previous_tower
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
