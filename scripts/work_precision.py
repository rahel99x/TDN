#!/usr/bin/env python3
"""Run the frozen training-free CPU work–precision screen in the project venv."""
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
    parser.add_argument("--config", type=Path, default=ROOT / "configs/work-precision.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run_dir, stage, report = None, None, None
    owned, exit_code = False, 1
    previous_tower = os.environ.get("TDN_TOWER_DIR")
    started = time.monotonic()
    try:
        from tdn.runtime.storage import configure_storage, contained_path
        from tdn.runtime.preflight import verify_runtime, execution_mode
        configure_storage()
        verify_runtime("cpu", "work-precision")
        from tdn.analysis.work_precision.protocol import (ARTIFACTS, digest,
            file_digest, make_protocol, validate_config)
        import yaml
        config_path = contained_path(args.config)
        config_fingerprint = file_digest(config_path)
        config = validate_config(yaml.safe_load(config_path.read_text()), smoke=args.smoke)
        candidate = contained_path(args.run_dir)
        if candidate.exists():
            raise ValueError("Work–precision screen requires a fresh nonexistent run directory; preserve prior results")
        import torch
        from tdn.runtime.metadata import software_metadata, write_json
        from tdn.runtime.precision import reference_precision
        from tdn.runtime.signal_handling import StopRequest
        from tdn.analysis.work_precision.experiment import plan, run, write_canonical
        torch.set_num_threads(config["intraop_threads"])
        torch.set_num_interop_threads(config["interop_threads"])
        reference_precision()
        software = software_metadata()
        command = [sys.executable, str(Path(__file__).resolve()),
                   *(sys.argv[1:] if argv is None else argv)]
        protocol = make_protocol(config, plan(config), software, command)
        # Reserve this run atomically. A competing same-path launch must fail
        # before it acquires any ownership or publishes an artifact.
        candidate.mkdir(parents=True, exist_ok=False)
        run_dir = candidate
        stage = {"stage": "work-precision", "status": "RUNNING", "device": "cpu",
                 "execution_mode": execution_mode(), "software": software,
                 "command": command, "protocol_sha256": digest(protocol),
                 "training_attempted": False, "training_performed": False, "actually_ran": False}
        write_json(run_dir / "protocol.json", protocol)
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "stage.json", stage)
        from tdn.reporting import attach_report, emit
        report, owned = attach_report(run_dir, name="tdn-work-precision",
            script="scripts/work_precision.py",
            parameters={"device": "cpu", "benchmark_suite": "work-precision",
                        "smoke": args.smoke, "config_sha256": digest(config),
                        "source_tree_sha256": software["source_tree_sha256"]},
            metadata={"protocol_sha256": digest(protocol), "training_attempted": False})
        os.environ["TDN_TOWER_DIR"] = str(report)
        stage["actually_ran"] = True
        write_json(run_dir / "stage.json", stage)

        def progress(name, result):
            print(f"TDN work–precision: {name}: {len(result['candidate_rows'])} retained candidates", flush=True)
            emit({"reported_candidates": len(result["candidate_rows"]),
                  "invalid_candidates": sum(row["status"] == "INVALID" for row in result["candidate_rows"])},
                 phase="work-precision")

        with StopRequest() as stop:
            result = run(protocol, run_dir, stop=stop, progress=progress)
        if result["status"] != "COMPLETED":
            stage["status"] = result["status"]
            raise RuntimeError(f"Work–precision screen {result['status']}; retained reports are not sealed")
        if software_metadata()["source_tree_sha256"] != software["source_tree_sha256"]:
            raise RuntimeError("Execution source changed during the screen; start a fresh run")
        if file_digest(config_path) != config_fingerprint:
            raise RuntimeError("Configuration changed during the screen; start a fresh run")
        manifest = {"version": 1, "protocol_sha256": digest(protocol),
                    "config_file_sha256": config_fingerprint,
                    "source_tree_sha256": software["source_tree_sha256"],
                    "files": {name: file_digest(run_dir / name) for name in ARTIFACTS}}
        write_json(run_dir / "manifest.json", manifest)
        (run_dir / "COMPLETED").write_text(file_digest(run_dir / "manifest.json") + "\n")
        stage.update(status="COMPLETED", scientific_outcome=result["scientific_outcome"],
                     elapsed_seconds=time.monotonic() - started)
        write_json(run_dir / "stage.json", stage)
        print((run_dir / "summary.txt").read_text(), end="")
        print(json.dumps({"status": "COMPLETED", "scientific_outcome": result["scientific_outcome"],
                          "report": str(run_dir / "summary.json")}, indent=2))
        exit_code = 0
    except (Exception, KeyboardInterrupt) as error:
        if stage is not None:
            if not (run_dir / "COMPLETED").exists():
                for name in ("summary.json", "work-precision.json"):
                    path = run_dir / name
                    if path.is_file():
                        payload = json.loads(path.read_text())
                        if payload.get("status") == "COMPLETED":
                            payload.update(status="INCOMPLETE", computational_status="INCOMPLETE",
                                           scientific_outcome="INCONCLUSIVE",
                                           sealing_error=f"{type(error).__name__}: {error}")
                            if name == "work-precision.json":
                                write_canonical(path, payload)
                            else:
                                write_json(path, payload)
                summary_text = run_dir / "summary.txt"
                if summary_text.is_file():
                    summary_text.write_text(summary_text.read_text() +
                        f"UNSEALED: {type(error).__name__}: {error}\n")
            failure_status = "INCOMPLETE" if stage.get("status") == "INCOMPLETE" else "FAILED"
            stage.update(status=failure_status, elapsed_seconds=time.monotonic() - started,
                         error=f"{type(error).__name__}: {error}")
            write_json(run_dir / "stage.json", stage)
        print(f"TDN work–precision: {type(error).__name__}: {error}", file=sys.stderr)
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 75 if stage and stage["status"] == "INCOMPLETE" else 1
    finally:
        try:
            if owned:
                from tdn.cli import finalize_tower_report
                tower_state = "INTERRUPTED" if stage["status"] == "INCOMPLETE" else stage["status"]
                if not finalize_tower_report(report, source_dirs=[run_dir], state=tower_state,
                        started=started, exit_code=exit_code,
                        metadata={"device": "cpu", "actually_ran": stage["actually_ran"],
                                  "training_attempted": False, "training_performed": False,
                                  "scientific_status": stage["status"]}):
                    exit_code = exit_code or 1
        finally:
            if previous_tower is None:
                os.environ.pop("TDN_TOWER_DIR", None)
            else:
                os.environ["TDN_TOWER_DIR"] = previous_tower
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
