#!/usr/bin/env python3
"""Read-only validation through the user's installed Tower Python API."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract")
    parser.add_argument("root")
    parser.add_argument("--max-bytes", type=int, choices=(8 << 20, 32 << 20), default=8 << 20)
    args = parser.parse_args(argv)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    # This project also has scripts/tower.py. Remove its import directory so
    # the installed Tower package, rather than that project helper, is loaded.
    own_directory = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != own_directory]
    try:
        from tower.artifacts import load_contract, validate_contract
    except ImportError:
        print("TDN: This interpreter cannot import your existing Tower installation. Set TDN_TOWER_PYTHON to its absolute Python path.", file=sys.stderr)
        return 2
    result = validate_contract(load_contract(args.contract), args.root, max_bytes=args.max_bytes)
    result["project_validation_scope"] = "Unmodified Tower API; explicit bounded consistency report allowance" if args.max_bytes == 32 << 20 else "Unmodified Tower API; ordinary bounded allowance"
    print(json.dumps(result, indent=2))
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
