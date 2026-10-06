#!/usr/bin/env python3
"""Run the frozen, training-free CPU mechanism audit in the project venv."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mechanism-audit.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run_dir, stage, report = None, None, None
    owned, exit_code = False, 1
    previous_tower = os.environ.get("TDN_TOWER_DIR")
    start = time.monotonic()
    try:
        # Identity, venv, storage and allocation precede scientific imports.
        from tdn.runtime.storage import configure_storage, contained_path
        from tdn.runtime.preflight import verify_runtime, execution_mode
        configure_storage()
        verify_runtime("cpu", "mechanism-audit")
        from tdn.analysis.mechanisms.protocol import (ARTIFACTS, PANELS, digest,
            file_digest, make_protocol, validate_config)
        import yaml
        config_path = contained_path(args.config)
        config_fingerprint = file_digest(config_path)
        config = validate_config(yaml.safe_load(config_path.read_text()), smoke=args.smoke)
        candidate = contained_path(args.run_dir)
        if candidate.exists() and (not candidate.is_dir() or any(candidate.iterdir())):
            raise ValueError("Mechanism audit requires an empty run directory; preserve prior results")
        import torch
        from tdn.runtime.metadata import software_metadata, write_json
        from tdn.runtime.precision import reference_precision
        from tdn.runtime.signal_handling import StopRequest
        from tdn.analysis.mechanisms.run import run
        torch.set_num_threads(config["intraop_threads"])
        torch.set_num_interop_threads(config["interop_threads"])
        reference_precision()
        software = software_metadata()
        modules = {name: importlib.import_module(f"tdn.analysis.mechanisms.{name}") for name in PANELS}
        plans = {name: modules[name].plan(config) for name in PANELS}
        command = [sys.executable, str(Path(__file__).resolve()),
                   *(sys.argv[1:] if argv is None else argv)]
        protocol = make_protocol(config, plans, software, command)
        candidate.mkdir(parents=True, exist_ok=True)
        run_dir = candidate
        stage = {"stage": "mechanism-audit", "status": "RUNNING", "device": "cpu",
                 "execution_mode": execution_mode(), "software": software,
                 "command": command, "protocol_sha256": digest(protocol), "actually_ran": False}
        write_json(run_dir / "protocol.json", protocol)
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "stage.json", stage)
        from tdn.reporting import attach_report, emit
        report, owned = attach_report(run_dir, name="tdn-mechanism-audit",
            script="scripts/mechanism_audit.py",
            parameters={"device": "cpu", "benchmark_suite": "mechanism-audit",
                        "smoke": args.smoke, "config_sha256": digest(config),
                        "source_tree_sha256": software["source_tree_sha256"]},
            metadata={"protocol_sha256": digest(protocol), "training_attempted": False})
        os.environ["TDN_TOWER_DIR"] = str(report)
        stage["actually_ran"] = True
        write_json(run_dir / "stage.json", stage)

        def progress(name, result):
            print(f"TDN mechanisms: {name}: {result['status']}; {len(result['rows'])} cases", flush=True)
            emit({"reported_cases": len(result["rows"]),
                  "correctness_failures": sum(row["kind"] == "correctness" and row["outcome"] == "FAIL"
                                              for row in result["rows"])}, phase=f"mechanisms-{name}")

        with StopRequest() as stop:
            result = run(protocol, run_dir, modules, stop=stop, progress=progress)
        if result["status"] != "COMPLETED":
            raise RuntimeError(f"Mechanism audit {result['status']}; retained reports are not sealed")
        if software_metadata()["source_tree_sha256"] != software["source_tree_sha256"]:
            raise RuntimeError("Execution source changed during the audit; start a fresh run")
        if file_digest(config_path) != config_fingerprint:
            raise RuntimeError("Configuration changed during the audit; start a fresh run")
        manifest = {"version": 1, "protocol_sha256": digest(protocol),
                    "config_file_sha256": config_fingerprint,
                    "source_tree_sha256": software["source_tree_sha256"],
                    "files": {name: file_digest(run_dir / name) for name in ARTIFACTS}}
        write_json(run_dir / "manifest.json", manifest)
        (run_dir / "COMPLETED").write_text(file_digest(run_dir / "manifest.json") + "\n")
        stage.update(status="COMPLETED", elapsed_seconds=time.monotonic() - start)
        write_json(run_dir / "stage.json", stage)
        print((run_dir / "summary.txt").read_text(), end="")
        print(json.dumps({"status": "COMPLETED", "report": str(run_dir / "summary.json")}, indent=2))
        exit_code = 0
    except (Exception, KeyboardInterrupt) as error:
        if stage is not None:
            # Completed calculations cannot claim a sealed successful audit
            # when provenance changed or final artifact publication failed.
            if not (run_dir / "COMPLETED").exists():
                for name in ("summary.json", "mechanism-audit.json"):
                    path = run_dir / name
                    if path.is_file():
                        payload = json.loads(path.read_text())
                        if payload.get("status") == "COMPLETED":
                            payload.update(status="INCOMPLETE", sealing_error=f"{type(error).__name__}: {error}")
                            write_json(path, payload)
                summary_text = run_dir / "summary.txt"
                if summary_text.is_file():
                    summary_text.write_text(summary_text.read_text() +
                        f"UNSEALED: {type(error).__name__}: {error}\n")
            stage.update(status="FAILED", elapsed_seconds=time.monotonic() - start,
                         error=f"{type(error).__name__}: {error}")
            write_json(run_dir / "stage.json", stage)
        print(f"TDN mechanisms: {type(error).__name__}: {error}", file=sys.stderr)
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        try:
            if owned:
                from tdn.cli import finalize_tower_report
                if not finalize_tower_report(report, source_dirs=[run_dir], state=stage["status"],
                        started=start, exit_code=exit_code,
                        metadata={"device": "cpu", "actually_ran": stage["actually_ran"],
                                  "training_attempted": False}):
                    exit_code = exit_code or 1
        finally:
            if previous_tower is None:
                os.environ.pop("TDN_TOWER_DIR", None)
            else:
                os.environ["TDN_TOWER_DIR"] = previous_tower
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
