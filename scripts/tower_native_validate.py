#!/usr/bin/env python3
"""Read-only validation through the user's installed Tower Python API."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys


def validate_roadmap_pages(root, load_contract, validate_contract):
    """Validate every indexed project page through the unchanged Tower API."""
    root = Path(root).resolve()
    result = {"validation": "all_indexed_roadmap_pages", "valid": False, "page_count": 0, "contracts": []}
    try:
        def confined(name, max_bytes=1 << 20):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Roadmap output index contains an unsafe path")
            path = root / relative
            for part in (path, *path.parents):
                if part == root:
                    break
                if part.is_symlink():
                    raise ValueError("Roadmap output index contains a symlink")
            if not path.is_file() or path.stat().st_size > max_bytes:
                raise ValueError("Roadmap output is missing or exceeds its read budget")
            return path
        index = json.loads(confined("outputs/roadmap-tables.json").read_text())
        if index.get("schema") != "tdn.roadmap.tower-tables/v1" or index.get("reporting_complete") is not True:
            raise ValueError("Roadmap reporting is incomplete or its index schema is unsupported")
        pages, counts = {}, Counter()
        for catalog in index["page_catalogs"]:
            path = confined(catalog["path"])
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != catalog["sha256"]:
                raise ValueError("Roadmap page catalog changed")
            entries = json.loads(raw)["pages"]
            if len(entries) != catalog["pages"] or len(entries) > 128:
                raise ValueError("Roadmap catalog has inconsistent page counts")
            for page in entries:
                if page["path"] in pages or len(pages) >= 131072:
                    raise ValueError("Roadmap page inventory is duplicated or too large")
                if type(page["rows"]) is not int or not 0 < page["rows"] <= 128:
                    raise ValueError("Roadmap page exceeds its row budget")
                pages[page["path"]] = page
                counts[page["table"]] += page["rows"]
        observed = {key: counts[key] for key in index["expected_rows"]}
        if (observed != index["expected_rows"] or observed != index["published_rows"]
                or set(counts) - set(observed) or len(pages) != index["page_count"]):
            raise ValueError("Roadmap page inventory does not cover every declared output row")
        checked, read_bytes = set(), 0
        for item in index["output_contracts"]:
            contract = load_contract(confined(item["path"], 256 << 10))
            if len(contract["outputs"]) != item["outputs"] or len(contract["outputs"]) > 128:
                raise ValueError("Roadmap page contract has inconsistent output counts")
            for output in contract["outputs"]:
                name = output["path"]
                if name not in pages or name in checked:
                    raise ValueError("Roadmap contracts omit or duplicate indexed pages")
                page = pages[name]
                raw = confined(name, 256 << 10).read_bytes()
                read_bytes += len(raw)
                if read_bytes > 4 << 30:
                    raise ValueError("Roadmap page read budget exceeded")
                if hashlib.sha256(raw).hexdigest() != page["sha256"] or len(raw) != page["bytes"]:
                    raise ValueError("Roadmap page differs from its recorded digest or byte count")
                if output.get("min_bytes") != len(raw) or output.get("max_bytes") != len(raw):
                    raise ValueError("Roadmap native contract byte count differs")
                if page["format"] == "csv" and (output.get("rows") != page["rows"] or output.get("format") != "csv"):
                    raise ValueError("Roadmap CSV contract row count differs")
                if page["table"] == "metrics":
                    from tower.metrics import _record
                    lines = raw.splitlines()
                    if len(lines) != page["rows"]:
                        raise ValueError("Roadmap canonical metric page has missing points")
                    for line in lines:
                        _record(json.loads(line))
                checked.add(name)
            native = validate_contract(contract, str(root), max_bytes=64 << 20, max_entries=4096)
            result["contracts"].append({"path": item["path"], "valid": native.get("valid"),
                "pages": len(contract["outputs"]), "failed_outputs": [entry for entry in native.get("outputs", [])
                    if entry.get("status") in ("failed", "incomplete")]})
            if not native.get("valid"):
                raise ValueError("Unmodified Tower rejected a roadmap page contract")
        if checked != set(pages):
            raise ValueError("Roadmap contracts do not cover all indexed pages")
        result.update(valid=True, page_count=len(pages), table_rows=observed, read_bytes=read_bytes)
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["error"] = str(error)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract")
    parser.add_argument("root")
    parser.add_argument("--max-bytes", type=int, choices=(8 << 20, 32 << 20, 64 << 20), default=8 << 20)
    parser.add_argument("--suite", choices=("agenda", "roadmap"))
    args = parser.parse_args(argv)
    if args.max_bytes == 64 << 20 and args.suite not in ("agenda", "roadmap"):
        parser.error("The 64 MiB allowance requires the named --suite agenda or roadmap budget")
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
    result["project_validation_scope"] = ("Unmodified Tower API; complete indexed roadmap page contracts"
        if args.suite == "roadmap" else "Unmodified Tower API; explicit bounded agenda report allowance"
        if args.max_bytes == 64 << 20 else "Unmodified Tower API; explicit bounded consistency report allowance"
        if args.max_bytes == 32 << 20 else "Unmodified Tower API; ordinary bounded allowance")
    if args.suite == "roadmap":
        result["roadmap"] = validate_roadmap_pages(args.root, load_contract, validate_contract)
        result["valid"] = bool(result.get("valid") and result["roadmap"]["valid"])
    print(json.dumps(result, indent=2))
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
