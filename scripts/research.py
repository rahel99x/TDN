#!/usr/bin/env python3
"""Bounded neural hypotheses, with a separate frozen-checkpoint GPU benchmark."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fresh_directory(value: Path) -> Path:
    from tdn.runtime.storage import contained_path
    lexical = value.absolute()
    for candidate in (lexical, *lexical.parents):
        if candidate.is_symlink():
            raise ValueError("Research output paths cannot traverse symlinks")
    path = contained_path(lexical)
    if path.exists():
        raise ValueError("Research requires a fresh run directory; preserve previous runs")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_manifest(run_dir, protocol, software, config_file_sha256):
    from tdn.research.protocol import digest, file_digest
    from tdn.runtime.metadata import write_json
    files = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in ("stage.json", "manifest.json", "COMPLETED") and ".partial" not in path.name:
            files[path.relative_to(run_dir).as_posix()] = file_digest(path)
    manifest = {"version": 1, "protocol_sha256": digest(protocol),
                "config_sha256": digest(protocol["config"]), "config_file_sha256": config_file_sha256,
                "source_tree_sha256": software["source_tree_sha256"],
                "software": {name: software[name] for name in ("python", "torch", "numpy", "scipy")},
                "files": files}
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "COMPLETED").write_text(file_digest(run_dir / "manifest.json") + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "benchmark"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, default=ROOT / "configs/research.yaml")
        sub.add_argument("--run-dir", type=Path, required=True)
        sub.add_argument("--device", choices=("cpu", "cuda"), default="cpu" if name == "run" else "cuda")
        if name == "run":
            sub.add_argument("--smoke", action="store_true")
        else:
            sub.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args(argv)
    stage = None
    run_dir = None
    tower_report, tower_owned, exit_code = None, False, 1
    previous_tower = os.environ.get("TDN_TOWER_DIR")
    start = time.monotonic()
    try:
        # Require project storage and a real CPU/Slurm task before importing torch.
        from tdn.runtime.storage import configure_storage, contained_path
        from tdn.runtime.preflight import execution_mode, verify_runtime
        configure_storage()
        verify_runtime("cpu", "research")
        from tdn.research.protocol import digest, file_digest, make_protocol, readable_summary, validate_config, verify_artifacts
        import yaml
        config_path = contained_path(args.config)
        raw_config = yaml.safe_load(config_path.read_text())
        if args.action == "run" and args.device != "cpu":
            raise ValueError("Bounded development training is CPU-only; GPU benchmark is a separate command")
        if args.action == "benchmark" and args.device != "cuda":
            raise ValueError("CPU timing is included in run; benchmark requires an allocated CUDA device")
        source_run = None
        if args.action == "benchmark":
            source_run = contained_path(args.source_run)
            source_manifest = verify_artifacts(source_run)
            protocol = json.loads((source_run / "protocol.json").read_text())
            config = validate_config(raw_config, smoke=protocol["smoke"])
            if config != protocol["config"] or file_digest(config_path) != source_manifest["config_file_sha256"]:
                raise ValueError("Benchmark configuration differs from the CPU source configuration")
            source_summary = json.loads((source_run / "summary.json").read_text())
            if source_summary.get("status") != "COMPLETED" or source_summary.get("device") != "cpu":
                raise ValueError("Benchmark requires a completed CPU research source")
        else:
            config = validate_config(raw_config, smoke=args.smoke)
        import torch
        from tdn.runtime.metadata import software_metadata, write_json
        from tdn.runtime.precision import reference_precision
        from tdn.runtime.signal_handling import StopRequest
        software = software_metadata()
        if source_run is not None:
            verify_artifacts(source_run, source_tree_sha256=software["source_tree_sha256"])
            if any(software[key] != expected for key, expected in source_manifest["software"].items()):
                raise ValueError("Software versions differ from the CPU source; generate a new CPU run")
        command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
        if source_run is None:
            protocol = make_protocol(config, smoke=args.smoke, software=software, command=command)
        run_dir = fresh_directory(args.run_dir)
        stage = {"stage": f"research-{args.action}", "status": "RUNNING", "device": args.device,
                 "execution_mode": execution_mode(), "software": software, "command": command,
                 "protocol_sha256": digest(protocol), "actually_ran": False}
        write_json(run_dir / "stage.json", stage)
        write_json(run_dir / "protocol.json", protocol)
        from tdn.reporting import attach_report, emit
        tower_parameters = {"action": args.action, "device": args.device, "smoke": protocol["smoke"],
                            "config_sha256": digest(config), "protocol_version": protocol["version"],
                            "source_tree_sha256": software["source_tree_sha256"],
                            "parent_plan_sha256": digest(protocol["parents"])}
        if source_run is not None:
            tower_parameters["source_manifest_sha256"] = file_digest(source_run / "manifest.json")
        tower_report, tower_owned = attach_report(
            run_dir, name=f"tdn-research-{args.action}", script="scripts/research.py",
            parameters=tower_parameters,
            metadata={"source_tree_sha256": software["source_tree_sha256"], "protocol_sha256": digest(protocol)})
        os.environ["TDN_TOWER_DIR"] = str(tower_report)
        emit({"started": 1}, phase=f"research-{args.action}")
        if source_run is not None and not source_summary["headroom"]["passed"]:
            result = {"status": "SKIPPED", "actually_ran": False, "device": "cuda",
                      "reason": "CPU classical headroom screen did not pass; no GPU numerical work performed",
                      "source_run": str(source_run)}
            write_json(run_dir / "summary.json", result)
            report = readable_summary(result)
            (run_dir / "summary.txt").write_text(report)
            stage.update(status="SKIPPED", elapsed_seconds=time.monotonic() - start)
            write_json(run_dir / "stage.json", stage)
            print(report, end="")
            print(json.dumps(result, indent=2))
            exit_code = 0
            return 0
        if args.device == "cuda":
            verify_runtime("cuda", "research-benchmark")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        reference_precision()
        from tdn.research.experiment import benchmark, run
        stage["actually_ran"] = True
        write_json(run_dir / "stage.json", stage)
        with StopRequest() as stop:
            result = run(protocol, run_dir, stop=stop) if source_run is None else benchmark(protocol, source_run, run_dir, stop=stop)
        if software_metadata()["source_tree_sha256"] != software["source_tree_sha256"]:
            raise RuntimeError("Research source changed during execution; completed marker withheld")
        if source_run is not None:
            verify_artifacts(source_run, source_tree_sha256=software["source_tree_sha256"])
        stage.update(status="COMPLETED", elapsed_seconds=time.monotonic() - start)
        write_json(run_dir / "stage.json", stage)
        report = readable_summary(result)
        (run_dir / "summary.txt").write_text(report)
        write_manifest(run_dir, protocol, software, file_digest(config_path))
        print(report, end="")
        print(json.dumps({"status": result["status"], "report": str(run_dir / "summary.json"),
                          "headroom_passed": result.get("headroom_passed")}, indent=2))
        exit_code = 0
        return 0
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, (InterruptedError, KeyboardInterrupt, TimeoutError))
        if stage is not None:
            stage.update(status="INTERRUPTED" if interrupted else "FAILED", error=f"{type(error).__name__}: {error}",
                         elapsed_seconds=time.monotonic() - start)
            write_json(run_dir / "stage.json", stage)
        print(f"TDN research: {type(error).__name__}: {error}", file=sys.stderr)
        exit_code = 75 if interrupted else 1
        return exit_code
    finally:
        try:
            if tower_owned:
                from tdn.cli import finalize_tower_report
                reported = finalize_tower_report(
                    tower_report, source_dirs=[run_dir], state=stage["status"],
                    started=start, exit_code=exit_code,
                    metadata={"device": args.device, "actually_ran": stage["actually_ran"]})
                if not reported and exit_code == 0:
                    return 1
        finally:
            if previous_tower is None:
                os.environ.pop("TDN_TOWER_DIR", None)
            else:
                os.environ["TDN_TOWER_DIR"] = previous_tower


if __name__ == "__main__":
    raise SystemExit(main())
