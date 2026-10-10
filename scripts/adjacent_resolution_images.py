#!/usr/bin/env python3
"""Export all declared grid/track checkpoint views after verified resolution aggregation.

CPU postprocessing only. No training, scheduler submission, or fresh accuracy
claim occurs here. The first declared seed/field/horizon is chosen before errors
are inspected. Every source run remains immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _workflow_api():
    import adjacent_workflow
    return adjacent_workflow


def resolve_run(value="latest"):
    from tdn.analysis.adjacent.prediction_view import _contained
    workflow = _workflow_api()
    supplied = Path(value)
    if value == "latest" or (not supplied.is_absolute() and "/" not in value):
        # Use the resolution launcher's own pointer and identifier validation.
        return _contained(workflow.workflow_path(value, resolution=True)).parent
    path = _contained(supplied)
    if path.is_file():
        if path.name != workflow.MANIFEST:
            raise ValueError("Choose a resolution run directory or its adjacent-workflow.json")
        return path.parent
    return path


def build_gallery_plan(run_dir):
    from tdn.analysis.adjacent.prediction_view import _contained, _coordinator, _verify_unit, load_bundle
    from tdn.research.protocol import file_digest
    run = _contained(run_dir)
    protocol, paths, coordinator_files = _coordinator(run)
    if protocol.get("study") != "adjacent-resolution":
        raise ValueError("This gallery requires a resolution-study workflow")
    aggregate_units = [name for name, unit in protocol["units"].items() if unit["kind"] == "resolution_aggregate"]
    if len(aggregate_units) != 1:
        raise ValueError("Exactly one complete resolution aggregate is required")
    sources = {str(path.relative_to(ROOT)): file_digest(path) for path in coordinator_files}
    verified = {}
    def visit(name):
        if name in verified:
            return
        manifest = _verify_unit(protocol, paths[name])
        for dependency in protocol["units"][name]["dependencies"]:
            visit(dependency)
            expected = file_digest(paths[dependency] / "science_manifest.json")
            if manifest["prerequisites"].get(dependency) != expected:
                raise ValueError("Resolution aggregate dependency lineage changed")
        verified[name] = manifest
        for relative, expected in manifest.get("artifacts", {}).items():
            sources[str((paths[name] / relative).relative_to(ROOT))] = expected
        for filename in ("science_manifest.json", "workflow-seal.json", "protocol.json"):
            path = paths[name] / filename
            sources[str(path.relative_to(ROOT))] = file_digest(path)
    visit(aggregate_units[0])
    if len({manifest["source_tree_sha256"] for manifest in verified.values()}) != 1:
        raise ValueError("Resolution aggregate dependencies disagree on scientific source")
    expected_evaluations = {name for name, unit in protocol["units"].items() if unit["kind"] == "resolution_evaluate"}
    if not expected_evaluations <= set(verified):
        raise ValueError("The aggregate does not cover every declared evaluation shard")
    cfg = protocol["resolution"]
    seed, horizon = int(cfg["seeds"][0]), float(cfg["evaluation_horizons"][0])
    selections = []
    for grid in cfg["grids"]:
        for track in cfg["tracks"]:
            parents = [index for name in expected_evaluations
                if protocol["units"][name]["grid"] == grid and protocol["units"][name]["track"] == track
                for index in protocol["units"][name]["field_indices"]]
            if not parents:
                raise ValueError("A declared grid/track lacks evaluation fields")
            choice = dict(family="channel_neural", track=track, grid=int(grid), train_grid=int(grid),
                          seed=seed, parent_index=min(parents), horizon=horizon)
            bundle = load_bundle(run, **choice)
            sources.update(bundle["provenance"]["source_artifacts"])
            selections.append(dict(**choice, parent_id=bundle["reference"]["parent_id"],
                model_id=bundle["spec"]["model_id"], checkpoint_sha256=bundle["spec"]["checkpoint_sha256"],
                selection_status=bundle["spec"]["selection_status"],
                name=f"n{grid}-{track}-channel-neural-seed{seed}"))
    return dict(schema="tdn.adjacent-resolution-images/v1", run_dir=str(run), profile=protocol["profile"],
        aggregate_unit=aggregate_units[0], verified_units=sorted(verified), source_artifacts=sources,
        selection_rule="All declared grids/tracks; first declared seed and horizon, smallest declared field index; chosen before errors",
        scope="CPU checkpoint postprocessing; single illustrative field per grid/track, no new superiority or native timing claim",
        selections=selections)


def export_gallery(plan, *, dpi=400):
    from tdn.analysis.adjacent.prediction_view import _contained, render_prediction_view
    from tdn.research.protocol import file_digest
    if type(dpi) is not int or not 400 <= dpi <= 600:
        raise ValueError("Use 400–600 DPI")
    run = _contained(plan["run_dir"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = _contained(ROOT / "outputs" / f"resolution-predictions-{run.name}-{stamp}")
    output.mkdir(parents=True, exist_ok=False)
    result = dict(plan, status="RUNNING", output_dir=str(output), dpi=dpi,
                  spatial_display="both", views=[])
    def save():
        (output / "image-gallery.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save()
    try:
        for selection in plan["selections"]:
            print(f"TDN resolution images: rendering {selection['name']} (raw + display smoothing)", flush=True)
            keywords = {key: selection[key] for key in
                ("family", "track", "grid", "train_grid", "seed", "parent_index", "horizon")}
            rendered = render_prediction_view(run, output / selection["name"], **keywords,
                                              dpi=dpi, spatial_display="both")
            result["views"].append(dict(selection=selection, **rendered))
            save()
            print(f"TDN resolution images: saved {rendered['output_dir']}", flush=True)
        for relative, expected in plan["source_artifacts"].items():
            if file_digest(_contained(ROOT / relative)) != expected:
                raise RuntimeError("A scientific source artifact changed during gallery generation")
        result.update(status="COMPLETED", files={str(path.relative_to(output)):file_digest(path)
            for path in sorted(output.rglob("*")) if path.is_file() and path.name != "image-gallery.json"})
    except Exception as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                      partial_outputs_retained=True)
        raise
    finally:
        save()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", help="Resolution run ID/path or latest (default)")
    parser.add_argument("--run", dest="run_option", help="Alternative named form of the run selector")
    parser.add_argument("--plan", action="store_true", help="Verify sources and print fixed selections without rendering")
    parser.add_argument("--dpi", type=int, default=400, help="400–600 DPI; vector PDF companions are always included")
    args = parser.parse_args(argv)
    if args.run and args.run_option:
        parser.error("Choose a positional run or --run, not both")
    for name, relative in {"TMPDIR":".runtime/tmp", "TMP":".runtime/tmp", "TEMP":".runtime/tmp",
                           "MPLCONFIGDIR":".cache/matplotlib", "XDG_CACHE_HOME":".cache"}.items():
        directory=ROOT/relative; directory.mkdir(parents=True, exist_ok=True)
        os.environ[name]=str(directory)
    os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    plan = build_gallery_plan(resolve_run(args.run_option or args.run or "latest"))
    if args.plan:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    result=export_gallery(plan, dpi=args.dpi)
    print(f"Resolution prediction gallery: {result['output_dir']}")
    for view in result["views"]:
        print(f"  {view['selection']['name']}: {view['output_dir']}")
    print("Each view includes raw and display-smoothed PNG/PDF panels, raw arrays, numerical metadata, and source hashes.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"TDN resolution images: {error}", file=sys.stderr)
        raise SystemExit(2)
