"""Replay the algebraic integration fixture; never fit the retained P1 bank."""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output == ROOT or not output.is_relative_to(ROOT):
        parser.error("Output must be a fresh directory inside this checkout")
    output.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import torch
    from tdn.analysis.response_preconditioner import regularization_workflow as audit
    from tdn.analysis.response_preconditioner.regularization_protocol import _fixture_config

    torch.set_num_threads(1)
    fixture = runpy.run_path(str(ROOT / "tests/test_response_preconditioner_regularization_workflow.py"))
    bank = fixture["SyntheticEvidence"]()
    config = _fixture_config()
    before = audit.legacy.source_snapshot()
    audit._json(output / "protocol.json", config)
    audit._json(output / "source.json", before)
    audit._json(output / "software.json", audit.legacy._software())
    summary = audit._study(output, config, evidence=bank, allow_fixture=True, plots=True)
    after = audit.legacy.source_snapshot()
    if before != after:
        raise RuntimeError("Source changed during fixture replay")
    audit._json(output / "source_after.json", after)
    summary.update(status="COMPLETED", scope="SYNTHETIC_ALGEBRAIC_IMPLEMENTATION_ONLY",
                   native_execution=False, source_fingerprint=before["fingerprint"])
    audit._json(output / "summary.json", summary)
    inputs = output / "synthetic_inputs"
    inputs.mkdir()
    for query_id, arrays in bank.arrays.items():
        np.savez_compressed(inputs / (query_id + ".npz"), **arrays)
    audit._json(inputs / "queries.json", bank.queries)
    audit._json(inputs / "access.json", bank.access)
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and not any(p.startswith(".") for p in path.relative_to(output).parts):
            files.append(dict(path=path.relative_to(output).as_posix(), bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    audit._json(output / "artifacts.json", dict(files=files, self_excluded="artifacts.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
