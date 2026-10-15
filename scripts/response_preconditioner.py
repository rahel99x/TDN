#!/usr/bin/env python3
"""Run bounded CPU verification or a nonpromoting fixed-feature development smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdn.analysis.response_preconditioner.workflow import execute


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "smoke"):
        child = subcommands.add_parser(command)
        child.add_argument("--device", choices=("cpu",), default="cpu")
        child.add_argument("--output", type=Path,
                           help="Fresh directory inside this checkout; generated under runs/ if omitted")
    args = parser.parse_args(argv)
    try:
        summary = execute(args.command, args.output, device=args.device)
    except (ValueError, OSError) as error:
        parser.exit(2, "TDN response preconditioner: "+str(error)+"\n")
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if summary["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
