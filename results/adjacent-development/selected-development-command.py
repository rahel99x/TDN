"""Execute only the declared 16x16 audit/D01/D05 development units."""
import json
from pathlib import Path
import subprocess
import sys
import time

from tdn.analysis.adjacent.protocol import build_protocol
from tdn.runtime.metadata import write_json

root = Path.cwd()
assert (root / "scripts/adjacent.py").is_file(), "Run from the project root"
run = root / "runs/adjacent-development-review-v2"
run.mkdir(exist_ok=False)
protocol = build_protocol("development")
selected = ["audit", "D01", "D05"]
write_json(run / "local-protocol.json", protocol)
write_json(run / "selected-units.json", {"units": selected,
    "scope": "Selected development diagnostics only; not the complete development DAG"})
states = {}
started = time.monotonic()
for stage in selected:
    command = [sys.executable, str(root / "scripts/adjacent.py"), "--stage", stage,
        "--profile", "development", "--run-dir", str(run / stage), "--device", "cpu",
        "--local-root", str(root)]
    if stage != "audit":
        command += ["--prerequisite-dir", f"audit={run / 'audit'}"]
    result = subprocess.run(command, cwd=root, check=False)
    states[stage] = "COMPLETED" if result.returncode == 0 else "FAILED"
    write_json(run / "local-state.json", states)
    if result.returncode:
        raise SystemExit(result.returncode)
write_json(run / "local-elapsed.json", {"elapsed_seconds": time.monotonic()-started,
    "scope": "Selected CPU CLI units including process startup, reports and sealing"})
print(json.dumps(states))
