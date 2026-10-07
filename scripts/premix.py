#!/usr/bin/env python3
"""Execute one immutable bounded premix stage in the project Python venv."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("accuracy", "scaling", "prepare", "neural"), required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--local-root", type=Path)
    args = parser.parse_args(argv)
    run_dir = report = stage = None
    owned, exit_code = False, 1
    previous_tower = os.environ.get("TDN_TOWER_DIR")
    started = time.monotonic()
    try:
        from tdn.runtime.storage import CARC_ROOT, configure_storage, contained_path
        from tdn.runtime.preflight import verify_runtime, execution_mode
        if args.local_root is not None:
            if args.local_root.resolve() != ROOT or ROOT == CARC_ROOT or any(
                    os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_STEP_ID", "TDN_EXECUTION_MODE")):
                raise ValueError("Local premix requires this CPU checkout without Slurm or desktop overrides")
            if args.device != "cpu":
                raise ValueError("Local premix is CPU-only; CUDA requires a real CARC allocation")
        if args.stage != "neural" and (args.device != "cpu" or args.dataset_dir is not None):
            raise ValueError("Only the neural stage accepts CUDA or a prepared dataset")
        if args.stage == "neural" and args.dataset_dir is None:
            raise ValueError("Neural stage requires its sealed prepared dataset")
        configure_storage()
        if execution_mode() not in ("local-cpu", "carc"):
            raise ValueError("Premix supports CARC and the separate local CPU workflow")
        verify_runtime(args.device, "premix-" + args.stage)
        candidate = contained_path(args.run_dir)
        if candidate.exists():
            raise ValueError("Preserve prior results: choose a fresh nonexistent run directory")
        from tdn.analysis.premix.artifacts import digest, file_digest, seal_stage, verify_stage
        from tdn.runtime.metadata import software_metadata, write_json
        from tdn.runtime.precision import reference_precision
        from tdn.runtime.signal_handling import StopRequest
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        reference_precision()
        software = software_metadata()
        if args.stage in ("accuracy", "scaling"):
            from tdn.analysis.premix import mechanisms as engine
            protocol = engine.build_protocol(profile=args.profile, panel=args.stage)
        else:
            from tdn.analysis.premix import neural as engine
            protocol = engine.build_protocol(profile=args.profile)
        dataset = None
        dataset_seal = None
        if args.dataset_dir is not None:
            dataset = contained_path(args.dataset_dir)
            dataset_seal = verify_stage(dataset, stage="prepare", profile=args.profile,
                                        source_tree_sha256=software["source_tree_sha256"])
            if dataset_seal["protocol_sha256"] != digest(protocol):
                raise ValueError("Prepared dataset has a different neural protocol")
        command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
        candidate.mkdir(parents=True, exist_ok=False)
        run_dir = candidate
        execution = {"stage": args.stage, "profile": args.profile, "device": args.device,
                     "execution_mode": execution_mode(), "command": command, "software": software,
                     "protocol_sha256": digest(protocol)}
        if os.environ.get("TDN_PREMIX_PROTOCOL_SHA256"):
            execution["workflow_protocol_sha256"] = os.environ["TDN_PREMIX_PROTOCOL_SHA256"]
        if dataset is not None:
            execution.update(dataset_dir=str(dataset), dataset_manifest_sha256=file_digest(dataset / "manifest.json"))
        write_json(run_dir / "execution.json", execution)
        write_json(run_dir / "protocol.json", protocol)
        stage = {**execution, "status": "RUNNING", "actually_ran": False,
                 "benchmark_suite": "premix", "training_attempted": args.stage == "neural"}
        write_json(run_dir / "stage.json", stage)
        from tdn.reporting import attach_report, emit
        report, owned = attach_report(run_dir, name="TDN/premix/" + args.stage,
            script="scripts/premix.py", parameters={"device": args.device, "stage": args.stage,
                "benchmark_suite": "premix", "profile": args.profile,
                "protocol_sha256": digest(protocol), "source_tree_sha256": software["source_tree_sha256"]})
        os.environ["TDN_TOWER_DIR"] = str(report)
        stage["actually_ran"] = True
        write_json(run_dir / "stage.json", stage)
        emit({"stage_started": 1}, phase="premix/" + args.stage)

        def progress(name, tables):
            print(f"TDN premix {args.stage}: {name}: {len(tables['candidate_rows'])} candidates", flush=True)
            emit({"candidate_rows": len(tables["candidate_rows"]),
                  "reference_rows": len(tables["reference_rows"])}, phase="premix/" + args.stage)

        with StopRequest() as stop:
            if args.stage in ("accuracy", "scaling"):
                result = engine.run(protocol, run_dir, stop=stop, progress=progress)
            elif args.stage == "prepare":
                result = engine.prepare(protocol, run_dir, stop=stop)
            else:
                result = engine.run(protocol, dataset, run_dir, device=args.device, stop=stop)
            if stop.requested:
                raise InterruptedError("Stop requested before stage sealing")
        if result.get("status") != "COMPLETED":
            if result.get("status") in ("INCOMPLETE", "INTERRUPTED"):
                raise InterruptedError("Premix stage stopped with partial evidence")
            raise RuntimeError("Premix engine did not complete; retained evidence is unsealed")
        if args.stage == "neural":
            from tdn.analysis.premix.comparison import summarize
            summarize(run_dir)
        if software_metadata()["source_tree_sha256"] != software["source_tree_sha256"]:
            raise RuntimeError("Execution source changed during premix stage; start a fresh run")
        if digest(json.loads((run_dir / "protocol.json").read_text())) != digest(protocol):
            raise RuntimeError("Premix protocol changed during execution")
        if dataset is not None:
            verify_stage(dataset, stage="prepare", profile=args.profile,
                         source_tree_sha256=software["source_tree_sha256"])
            if file_digest(dataset / "manifest.json") != execution["dataset_manifest_sha256"]:
                raise RuntimeError("Prepared dataset changed during training")
        seal_stage(run_dir, stage=args.stage, profile=args.profile,
                   source_tree_sha256=software["source_tree_sha256"])
        stage.update(status="COMPLETED", elapsed_seconds=time.monotonic() - started)
        write_json(run_dir / "stage.json", stage)
        emit({"stage_completed": 1}, phase="premix/" + args.stage)
        if (run_dir / "summary.txt").is_file():
            print((run_dir / "summary.txt").read_text(), end="")
        print(json.dumps({"status": "COMPLETED", "stage": args.stage,
                          "summary": str(run_dir / "summary.json"), "tower": str(report)}, indent=2))
        exit_code = 0
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, (InterruptedError, TimeoutError, KeyboardInterrupt))
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 75 if interrupted else 1
        if stage is not None:
            stage.update(status="INTERRUPTED" if interrupted else "FAILED",
                         error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
            write_json(run_dir / "stage.json", stage)
            if not (run_dir / "COMPLETED").exists():
                summary_path = run_dir / "summary.json"
                summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
                summary.update(status=stage["status"], scientific_outcome="INCONCLUSIVE",
                               sealing_error=stage["error"], stage=args.stage, device=args.device)
                write_json(summary_path, summary)
        print(f"TDN premix: {type(error).__name__}: {error}", file=sys.stderr)
    finally:
        try:
            if owned:
                from tdn.cli import finalize_tower_report
                if not finalize_tower_report(report, source_dirs=[run_dir], state=stage["status"],
                        started=started, exit_code=exit_code,
                        metadata={"device": args.device, "actually_ran": stage["actually_ran"],
                                  "stage": args.stage, "scientific_status": stage["status"]}):
                    exit_code = exit_code or 1
        finally:
            if previous_tower is None:
                os.environ.pop("TDN_TOWER_DIR", None)
            else:
                os.environ["TDN_TOWER_DIR"] = previous_tower
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
