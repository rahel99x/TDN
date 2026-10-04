"""Executable stages with contained outputs, real exit status and factual reports."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from tdn.config import config_hash, load_config
from tdn.runtime.storage import configure_storage, contained_path
from tdn.runtime.metadata import software_metadata, write_json


def finalize_tower_report(report, *, source_dirs, state, started, exit_code, metadata=None):
    """Publish a separate reporting view without changing scientific evidence.

    Return False on a reporting failure so successful science cannot be
    mistaken for a complete reporting export. Callers retain nonzero exits.
    """
    try:
        from tdn.reporting import finish_report
        from tdn.tower_analytics import publish_outputs
        results = publish_outputs(report, source_dirs)
        terminal = "INTERRUPTED" if state.startswith("PAUSED") else state
        # A withheld GPU benchmark is a completed decision, not GPU work.
        if terminal == "SKIPPED":
            terminal = "COMPLETED"
        finish_report(report, state=terminal, runtime_seconds=time.monotonic()-started,
                      exit_code=exit_code, results=results,
                      metadata={"scientific_status": state, **(metadata or {})})
        print(f"TDN Tower report: {report}")
        return True
    except Exception as error:
        print(f"TDN Tower reporting failed: {type(error).__name__}: {error}", file=sys.stderr)
        return False

def parser():
    p=argparse.ArgumentParser(prog="tdn")
    p.add_argument("stage", choices=["audit","generate","train","evaluate","benchmark","compare","calibrate","validate-config"])
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--dataset",type=Path)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--resume",default="none",help="Training resume or the single selected evaluation checkpoint")
    p.add_argument("--gate-report",type=Path,default=os.environ.get("TDN_GATE_REPORT"))
    p.add_argument("--calibration-report",type=Path,default=os.environ.get("TDN_CALIBRATION_REPORT"))
    return p

def require_report(path,config,key):
    if path is None: raise ValueError(f"Required {key} report is missing; run prerequisite stage")
    report=json.loads(contained_path(path).read_text())
    stage_path=Path(path).parent/"stage.json"
    if not stage_path.is_file(): raise ValueError(f"{key} requires its completed measured stage.json")
    stage=json.loads(stage_path.read_text())
    if stage.get("status") != "COMPLETED" or not stage.get("actually_ran"):
        raise ValueError(f"{key} prerequisite did not complete")
    if stage.get("software",{}).get("source_tree_sha256") != software_metadata()["source_tree_sha256"]:
        raise ValueError(f"{key} prerequisite is stale after source changes; rerun it")
    if report.get("config_hash") != config_hash(config): raise ValueError(f"{key} configuration hash does not match")
    if key == "calibration":
        if not report.get("passed"): raise ValueError("Required GPU calibration has not passed")
    else:
        gates=report.get("gates",{})
        for gate in ("G1","G2","G3","G4"):
            item=gates.get(gate)
            passed=item.get("passed",False) if isinstance(item,dict) else item is True
            if not passed: raise ValueError(f"{gate} prevents the pilot: {item}")
    return report

def main(argv=None):
    args=parser().parse_args(argv)
    root=configure_storage()
    config=load_config(contained_path(args.config))
    run=contained_path(args.run_dir);run.mkdir(parents=True,exist_ok=True)
    if (run/"COMPLETED").exists():
        raise SystemExit("Run directory already completed; select a new directory to preserve results")
    data=contained_path(args.dataset or root/"datasets"/config["experiment"])
    checkpoint=None if args.resume == "none" else contained_path(args.resume)
    start=time.monotonic()
    status={"stage":args.stage,"status":"RUNNING","config_hash":config_hash(config),"command":[sys.executable,"-m","tdn.cli",*(argv or sys.argv[1:])],
            "actually_ran":True,"scope":"newly measured development run", "device":args.device,
            "software":software_metadata(),"project_root":str(root)}
    write_json(run/"stage.json",status);write_json(run/"config.json",config)
    tower_report, tower_owned, exit_code = None, False, 1
    previous_tower = os.environ.get("TDN_TOWER_DIR")
    try:
        from tdn.reporting import attach_report, emit
        tower_parameters = {"stage": args.stage, "device": args.device,
                            "config_sha256": config_hash(config), "experiment": config["experiment"],
                            "source_tree_sha256": status["software"]["source_tree_sha256"]}
        if args.stage in ("train", "evaluate", "benchmark", "compare") and (data / "manifest.json").is_file():
            tower_parameters["dataset_manifest_sha256"] = hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
        tower_report, tower_owned = attach_report(
            run, name=f"tdn-{args.stage}", script="tdn/cli.py",
            parameters=tower_parameters,
            metadata={"source_tree_sha256": status["software"]["source_tree_sha256"]})
        os.environ["TDN_TOWER_DIR"] = str(tower_report)
        emit({"started": 1}, phase=args.stage, completed=0, total=1, unit="stages")
        import torch
        torch.set_num_threads(config["runtime"]["intraop_threads"])
        torch.set_num_interop_threads(config["runtime"]["interop_threads"])
        from tdn.runtime.precision import reference_precision
        reference_precision()
        from tdn.runtime.preflight import verify_runtime
        verify_runtime(args.device,args.stage)
        if args.stage in ("generate","train","compare") and config["validation"]["require_headroom"]:
            require_report(args.gate_report,config,"audit")
        if args.stage == "train" and (config["precision"]["network_autocast"] != "none" or config["precision"]["compile_mode"] != "eager"):
            require_report(args.calibration_report,config,"calibration")
            os.environ["TDN_CALIBRATION_REPORT"]=str(contained_path(args.calibration_report))
        if args.stage == "audit":
            from tdn.analysis import audit
            result=audit(config,run,args.device)
            result.setdefault("config_hash",config_hash(config));write_json(run/"audit.json",result)
            if config["validation"]["require_headroom"]:
                for gate in ("G1","G2","G3","G4"):
                    if not result.get("gates",{}).get(gate,{}).get("passed",False):
                        raise ValueError(f"{gate} failed empirical screening; results retained and pilot dependencies stop")
        elif args.stage == "generate":
            from tdn.data import generate_dataset
            result={"manifest":str(generate_dataset(config,data))}
        elif args.stage == "train":
            from tdn.train import train
            result=train(config,data,run,device=args.device,resume=checkpoint)
            if result.get("status") == "PAUSED_NEEDS_RESUME":
                status.update(status="PAUSED_NEEDS_RESUME",result=result);write_json(run/"stage.json",status)
                raise SystemExit(75)
        elif args.stage in ("evaluate","benchmark"):
            if checkpoint is None: raise ValueError("Provide --resume with one validation-selected best checkpoint for all cases")
            from tdn.analysis import evaluate,benchmark
            result=(evaluate if args.stage == "evaluate" else benchmark)(config,data,checkpoint,run,device=args.device)
        elif args.stage == "calibrate":
            from tdn.runtime.calibration import calibrate
            result=calibrate(config,run,args.device)
        elif args.stage == "compare":
            from tdn.analysis.learned_controls import compare_learned_controls
            result=compare_learned_controls(config,data,run/"controls",device=args.device)
            if result.get("status") != "COMPLETE":
                raise RuntimeError("Learned-control screen did not complete all families; failure reports retained")
        else: result={"valid":True}
        status.update(status="COMPLETED",elapsed_seconds=time.monotonic()-start,result=result)
        write_json(run/"stage.json",status)
        (run/"COMPLETED").write_text(config_hash(config)+"\n")
        with (root/"WORK_LOG.md").open("a") as log:
            log.write(f"\n- Executed `{args.stage}`; config `{config_hash(config)}`; commit `{status['software']['git_commit']}`; report `{run.relative_to(root)}/stage.json`; status COMPLETED, {status['elapsed_seconds']:.3f} seconds.\n")
        print(json.dumps({"status":"COMPLETED","stage":args.stage,"report":str(run/"stage.json")},indent=2))
        exit_code = 0
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
        status.update(status="PAUSED_NEEDS_RESUME" if e.code == 75 else "FAILED",elapsed_seconds=time.monotonic()-start)
        write_json(run/"stage.json",status)
        raise
    except KeyboardInterrupt:
        exit_code = 130
        status.update(status="INTERRUPTED", error="KeyboardInterrupt",
                      elapsed_seconds=time.monotonic()-start)
        write_json(run/"stage.json",status)
        raise
    except Exception as e:
        status.update(status="FAILED",error=f"{type(e).__name__}: {e}",elapsed_seconds=time.monotonic()-start)
        write_json(run/"stage.json",status)
        print(status["error"],file=sys.stderr)
        return 1
    finally:
        try:
            if tower_owned:
                sources = [run]
                if args.stage in ("generate", "train", "evaluate", "benchmark", "compare") and data.is_dir():
                    sources.append(data)
                reported = finalize_tower_report(
                    tower_report, source_dirs=sources, state=status["status"],
                    started=start, exit_code=exit_code,
                    metadata={"device": args.device, "actually_ran": status["actually_ran"]})
                if not reported and exit_code == 0:
                    return 1
        finally:
            if previous_tower is None:
                os.environ.pop("TDN_TOWER_DIR", None)
            else:
                os.environ["TDN_TOWER_DIR"] = previous_tower
    return 0

if __name__ == "__main__": sys.exit(main())
