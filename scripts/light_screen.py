#!/usr/bin/env python3
"""Run the predeclared CPU screen inside an allocation and retain its evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def interrupted(signum, frame):
    raise InterruptedError(f"Light screen interrupted by signal {signum}; start a fresh run")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    stage = None
    run_dir = None
    previous_handlers = {}
    tower_report, tower_owned, exit_code = None, False, 1
    previous_tower = os.environ.get("TDN_TOWER_DIR")
    start = time.monotonic()
    try:
        # Storage and allocation checks precede scientific imports or outputs.
        from tdn.runtime.storage import configure_storage, contained_path
        from tdn.runtime.preflight import execution_mode, verify_runtime

        configure_storage()
        verify_runtime("cpu", "light-screen")
        from tdn.config import config_hash, load_config
        from tdn.analysis.light_screen import run, validate_light_config
        from tdn.runtime.metadata import software_metadata, write_json
        from tdn.runtime.precision import reference_precision
        import torch

        config = validate_light_config(load_config(contained_path(args.config)))
        run_dir = contained_path(args.run_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError("Light screen requires an empty run directory; preserve prior results")
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(config["runtime"]["intraop_threads"])
        torch.set_num_interop_threads(config["runtime"]["interop_threads"])
        reference_precision()
        stage = {"stage": "light-screen", "status": "RUNNING", "actually_ran": True,
                 "device": "cpu", "execution_mode": execution_mode(),
                 "config_hash": config_hash(config), "software": software_metadata(),
                 "command": [sys.executable, str(Path(__file__).resolve()),
                             "--config", str(contained_path(args.config)),
                             "--run-dir", str(run_dir)],
                 "scope": "Bounded CPU scientific screening; no training or pilot authorization"}
        write_json(run_dir / "stage.json", stage)
        write_json(run_dir / "config.json", config)
        from tdn.reporting import attach_report, emit
        tower_report, tower_owned = attach_report(
            run_dir, name="tdn-light-screen", script="scripts/light_screen.py",
            parameters={"device": "cpu", "config_sha256": config_hash(config),
                        "source_tree_sha256": stage["software"]["source_tree_sha256"]},
            metadata={"source_tree_sha256": stage["software"]["source_tree_sha256"]})
        os.environ["TDN_TOWER_DIR"] = str(tower_report)
        emit({"started": 1}, phase="light-screen", completed=0, total=6, unit="cases")
        for name in ("SIGUSR1", "SIGTERM"):
            if hasattr(signal, name):
                signum = getattr(signal, name)
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, interrupted)
        result = run(config, run_dir, device="cpu")
        if result.get("status") != "COMPLETED":
            raise RuntimeError("Scientific screen did not finish its declared case budget; results retained")
        if software_metadata()["source_tree_sha256"] != stage["software"]["source_tree_sha256"]:
            raise RuntimeError("Execution source changed during screening; results are incomplete")
        stage.update(status="COMPLETED", elapsed_seconds=time.monotonic() - start,
                     result={"summary": str(run_dir / "summary.json"),
                             "readable_summary": str(run_dir / "summary.txt")})
        write_json(run_dir / "stage.json", stage)
        (run_dir / "COMPLETED").write_text(config_hash(config) + "\n")
        print((run_dir / "summary.txt").read_text())
        print(json.dumps({"status": "COMPLETED", "stage": "light-screen",
                          "report": str(run_dir / "summary.json")}, indent=2))
        exit_code = 0
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if stage is not None:
            stage.update(status="FAILED", elapsed_seconds=time.monotonic() - start,
                         error=f"{type(error).__name__}: {error}")
            write_json(run_dir / "stage.json", stage)
        print(f"TDN light screen: {type(error).__name__}: {error}", file=sys.stderr)
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
        return exit_code
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        try:
            if tower_owned:
                from tdn.cli import finalize_tower_report
                reported = finalize_tower_report(
                    tower_report, source_dirs=[run_dir], state=stage["status"],
                    started=start, exit_code=exit_code,
                    metadata={"device": "cpu", "actually_ran": stage["actually_ran"]})
                if not reported and exit_code == 0:
                    return 1
        finally:
            if previous_tower is None:
                os.environ.pop("TDN_TOWER_DIR", None)
            else:
                os.environ["TDN_TOWER_DIR"] = previous_tower


if __name__ == "__main__":
    sys.exit(main())
