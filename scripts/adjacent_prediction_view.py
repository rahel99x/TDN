#!/usr/bin/env python3
"""Render a verified frozen adjacent prediction without changing its run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable, relative in {"TMPDIR": ".runtime/tmp", "TMP": ".runtime/tmp", "TEMP": ".runtime/tmp",
        "MPLCONFIGDIR": ".cache/matplotlib", "XDG_CACHE_HOME": ".cache"}.items():
    directory = ROOT / relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Verified adjacent coordinator run under this project")
    parser.add_argument("--output-dir", required=True, help="New directory outside the source run, under this project")
    parser.add_argument("--family", choices=("channel_neural", "channel_global", "channel_affine", "channel_fixed"), default="channel_neural")
    parser.add_argument("--track", choices=("discrete", "continuum"), default="discrete")
    parser.add_argument("--parent-index", type=int, default=0)
    parser.add_argument("--horizon", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--grid", type=int, choices=(64, 128),
                        help="Resolution-study evaluation grid; defaults to the minimum declared grid")
    parser.add_argument("--train-grid", type=int, choices=(64, 128),
                        help="Resolution-study training grid; defaults to --grid; 64→128 shows transfer")
    parser.add_argument("--spatial-display", choices=("raw", "bicubic", "both"), default="raw",
                        help="Display only; raw arrays and metrics never change. Both retains raw and labeled smoothed companions.")
    parser.add_argument("--dpi", type=int, default=400, help="400–600 DPI; PDFs retain vector labels")
    args = parser.parse_args(argv)
    from tdn.analysis.adjacent.prediction_view import render_prediction_view
    result = render_prediction_view(**vars(args))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"TDN prediction view: {error}", file=sys.stderr)
        raise SystemExit(2)
