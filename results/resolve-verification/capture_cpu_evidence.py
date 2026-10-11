"""Export exact JSON evidence from completed local Resolve runs, not checkpoints.

Run from the repository root after smoke/development completion. Original run
directories remain intact. The atlas is a separate display projection.
"""
from pathlib import Path
import gzip
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tdn.analysis.resolve.engine import verify_science

NAMES = ("summary.json", "protocol.json", "science_manifest.json", "rows.json",
         "fit_results.json", "fit_endpoints.json", "fit_basis_cases.json",
         "interaction_rows.json", "interaction_cases.json", "interaction_summary.json",
         "fairness_results.json", "fairness_metrics.json", "learning_curves.json",
         "kernel_rows.json", "profiling.json", "memory_summary.json", "memory_selection.json")


def capture(run):
    run = run.resolve()
    if not run.is_relative_to(ROOT / "runs"):
        raise ValueError("Only project-contained run sources are accepted")
    protocol = json.loads((run / "local-protocol.json").read_text())
    records, stages, tower = {}, {}, []
    for stage in protocol["units"]:
        base = run / stage
        seal = verify_science(protocol, base)
        stages[stage] = {"source_tree_sha256": seal["source_tree_sha256"],
            "science_manifest_sha256": hashlib.sha256((base / "science_manifest.json").read_bytes()).hexdigest(),
            "summary": json.loads((base / "summary.json").read_text())}
        for name in NAMES:
            path = base / name
            if path.exists():
                raw = path.read_bytes()
                records[f"{stage}/{name}"] = dict(sha256=hashlib.sha256(raw).hexdigest(),
                    bytes=len(raw), document=json.loads(raw))
    for path in sorted(run.glob("*-tower/*/outputs/frontier-results.json")):
        value = json.loads(path.read_text())
        tower.append(dict(path=str(path.relative_to(run)),
            reporting_complete=value["reporting_complete"], omissions=value["reporting_omission_count"],
            seal_statuses=[s["seal_status"] for s in value["sources"]]))
    return dict(schema="tdn.resolve-cpu-verification/v1", run_id=run.name,
        scope="Cloud CPU development; no actual CUDA, native Slurm or confirmation claim",
        protocol=protocol, stages=stages, tower=tower, exact_json_documents=records,
        omitted="Binary checkpoints/teacher arrays and full memory trajectories remain in original run; this is an evidence excerpt, not a resumable archive")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        data = capture(Path(arg))
        target = Path(__file__).parent / f"{data['run_id']}-evidence.json.gz"
        with gzip.open(target, "wt", encoding="utf-8", compresslevel=5) as stream:
            json.dump(data, stream, separators=(",", ":"), allow_nan=False)
        print(target, target.stat().st_size)
