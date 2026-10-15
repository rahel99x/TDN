#!/usr/bin/env python3
"""Verify R7 implementation, or run its frozen study within native Raven/Slurm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "study"):
        child = commands.add_parser(command, allow_abbrev=False)
        child.add_argument("--device", choices=("cpu",), default="cpu")
        child.add_argument("--output", type=Path,
                           help="Fresh project-contained output; native study uses the Raven attempt")
        if command == "study":
            child.add_argument("--contract", required=True, help="Prepared native source/software contract")
    args = parser.parse_args(argv)
    try:
        if args.command == "study":
            from response_preconditioner_r7_native import require_study_context
            require_study_context(args.contract, args.output)
        # Import after native authorization: no scientific work during preparation.
        from tdn.analysis.response_preconditioner.r7_workflow import execute
        summary = execute(args.command, args.output, device=args.device)
    except (ValueError, OSError) as error:
        parser.exit(2, "TDN R7: " + str(error) + "\n")
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if summary["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
