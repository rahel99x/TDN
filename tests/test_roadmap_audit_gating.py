"""Scientific candidate failures remain reviewable without aborting other work."""
import json

import pytest

from tdn.analysis.roadmap import engine, numerical_audit
from tdn.analysis.roadmap.core import check
from tdn.analysis.roadmap.protocol import build_protocol


@pytest.mark.parametrize("category,blocks", [("math", False), ("gap", False), ("correctness", True)])
def test_scientific_bad_does_not_block_but_structural_failure_does(tmp_path, monkeypatch, category, blocks):
    monkeypatch.setattr(engine, "software_metadata", lambda: {"source_tree_sha256": "8" * 64})
    def measured_failure(ctx):
        ctx.record("deliberate-failure", ["M21"], checks=[
            check("candidate-interval" if category != "correctness" else "exact-algebraic-null", 1., 0., category=category)])
        return {"finite_diagnostic": True}
    monkeypatch.setattr(numerical_audit, "run", measured_failure)
    path = tmp_path / category
    if blocks:
        with pytest.raises(RuntimeError, match="structural correctness"):
            engine.run_stage(build_protocol("smoke"), "audit", path)
        assert not (path / "COMPLETED").exists()
    else:
        result = engine.run_stage(build_protocol("smoke"), "audit", path)
        assert result["status"] == "COMPLETED" and result["correctness_failures"] == 0
        engine.verify_science(build_protocol("smoke"), path)
    rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines()]
    assert rows[0]["checks"][0]["verdict"] == "BAD"
    assert rows[0]["assessment"]["verdict"] == "BAD"
