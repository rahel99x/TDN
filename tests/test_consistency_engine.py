"""Fresh-parent isolation, immutable data, checkpoint freeze and failure retention."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.consistency import engine
from tdn.analysis.consistency.protocol import DEVELOPMENT_REGIMES, expected_case_ids, expected_required_case_ids


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    path = tmp_path_factory.mktemp("consistency-data")
    protocol = engine.build_protocol("smoke")
    summary = engine.prepare(protocol, path)
    yield protocol, path, summary
    torch.set_num_threads(original_threads)


def test_full_protocol_fresh_and_frozen_with_broader_development():
    protocol = engine.build_protocol("full")
    engine.validate_protocol(protocol)
    assert [sum(p["split"] == s for p in protocol["parents"]) for s in
            ("train", "validation", "diagnostic")] == [48, 24, 48]
    assert protocol["seeds"] == [940011, 940021, 940031]
    assert protocol["updates"] == 300
    assert len(expected_case_ids()) == 128
    assert len(expected_required_case_ids()) == 110
    assert protocol["audit_case_ids"] == expected_case_ids()
    assert len({p["seed"] for p in protocol["parents"]}) == 120
    assert all(p["seed"] >= 910000 for p in protocol["parents"])
    for split, horizons in (("train", [.04, .08, .16, .24]), ("validation", [.06, .12, .20])):
        parents = [p for p in protocol["parents"] if p["split"] == split]
        assert {p["regime"] for p in parents} == set(DEVELOPMENT_REGIMES)
        assert len({p["mean"] for p in parents}) > 6
        assert {p["reaction_rate"] for p in parents} == {.5, 2., 6.}
        assert {p["kappa"] for p in parents} == {.001, .003, .009}
        for parent in parents:
            assert set(parent["horizons"]) == set(horizons + [2 * h for h in horizons])
            for row in parent["dimensionless"]:
                assert row["reaction_h"] == parent["reaction_rate"] * row["horizon"]
                assert row["diffusion_h_over_dx2"] == parent["kappa"] * row["horizon"] * parent["grid"][0]**2
    for field, value in (("updates", 301), ("families", ["premix"]), ("reference_tolerance", 1e-4)):
        with pytest.raises(ValueError, match="immutable"):
            engine.validate_protocol({**protocol, field: value})
    with pytest.raises(ValueError):
        engine.build_protocol("larger")


def test_protocol_controller_import_does_not_import_torch():
    result = subprocess.run([sys.executable, "-c", "import sys; from tdn.analysis.consistency.protocol import build_protocol; build_protocol('full'); assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_lazy_analysis_public_functions_retain_identity():
    import tdn.analysis as analysis
    from tdn.analysis import workflow
    assert analysis.audit is workflow.audit
    assert analysis.benchmark is workflow.benchmark
    assert analysis.evaluate is workflow.evaluate
    with pytest.raises(AttributeError):
        getattr(analysis, "missing_public_function")


def test_preparation_uses_fresh_accepted_fp64_teachers_and_train_only_fit(prepared):
    protocol, path, summary = prepared
    assert summary["counts"] == {"parents": 24, "references": 78, "accepted_references": 78}
    dataset, normalization = engine._load_dataset(protocol, path)
    assert dataset["schema"] == "tdn.consistency-neural/v1"
    assert len({p["initial_state_sha256"] for p in dataset["parents"]}) == 24
    assert all(p["initial"].dtype == torch.float64 for p in dataset["parents"])
    train = [p for p in dataset["parents"] if p["split"] == "train"]
    fitted = engine.fit_feature_normalization(((p["initial"], engine.Equation(p["kappa"], p["reaction_rate"]),
        engine.Geometry(tuple(p["grid"]), (1., 1.))) for p in train), t_ref=protocol["t_ref"], U_ref=protocol["U_ref"])
    assert torch.equal(normalization[0], fitted[0].float())
    assert torch.equal(normalization[1], fitted[1].float())
    for p in train:
        assert float(p["initial"].mean()) == pytest.approx(p["mean"], abs=1e-14)
        assert float((p["initial"] - p["mean"]).abs().max()) == pytest.approx(p["amplitude"], abs=1e-14)
    with pytest.raises(FileExistsError, match="fresh"):
        engine.prepare(protocol, path)


def test_byte_tampering_rejected_before_deserialization(prepared, tmp_path, monkeypatch):
    protocol, source, _ = prepared
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    with (tmp_path / "dataset.pt").open("ab") as handle:
        handle.write(b"changed")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Unverified artifact must not deserialize"))
    with pytest.raises(ValueError, match="integrity"):
        engine._load_dataset(protocol, tmp_path)


def test_validation_fit_cannot_be_relabelled_as_training(prepared, tmp_path):
    protocol, source, _ = prepared
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    normalization = json.loads((tmp_path / "normalization.json").read_text())
    normalization["parent_ids"] = [p["parent_id"] for p in protocol["parents"] if p["split"] == "validation"]
    engine.write_json(tmp_path / "normalization.json", normalization)
    names = list(json.loads((tmp_path / "dataset_manifest.json").read_text())["artifacts"])
    engine._manifest(tmp_path, names, protocol, "dataset_manifest.json")
    with pytest.raises(ValueError, match="training-only"):
        engine._load_dataset(protocol, tmp_path)


def test_training_freezes_all_seven_arms_before_any_diagnostics(prepared, tmp_path, monkeypatch):
    protocol, source, _ = prepared
    original = engine._evaluate
    inspections = []
    def check(protocol, dataset, models, records, path, device, budget, candidates):
        frozen = json.loads((path / "checkpoint_freeze.json").read_text())
        assert frozen["all_training_completed_before_diagnostics"] is True
        assert len(records) == 7 and len(frozen["artifacts"]) == 7
        assert all(record["steps"] == 2 for record in records)
        assert len(json.loads((path / "consistency.json").read_text())["rows"]) == 28
        assert all(engine.file_digest(path / name) == value for name, value in frozen["artifacts"].items())
        inspections.append(True)
        return original(protocol, dataset, models, records, path, device, budget, candidates)
    monkeypatch.setattr(engine, "_evaluate", check)
    summary = engine.run(protocol, source, tmp_path)
    assert inspections == [True]
    assert summary["actual_neural_training"] is True
    assert summary["fnopaper_reproduction"] is False
    assert summary["counts"]["training_records"] == 7
    assert summary["counts"]["candidates"] == 240
    records = json.loads((tmp_path / "training.json").read_text())["rows"]
    assert len({r["sample_schedule_sha256"] for r in records}) == 1
    assert all(r["diagnostics_seen_during_training"] is False for r in records)
    for row in records:
        finite = [point for point in row["history"] if point["validation_loss"] is not None]
        assert row["selected_step"] == min(finite, key=lambda point: point["validation_loss"])["step"]
        if row["selected_step"] == 0:
            assert row["selection"] == "SELECTED_INITIALIZATION"
    candidates = json.loads((tmp_path / "candidates.json").read_text())["rows"]
    assert {r["family"] for r in candidates} == set(engine.FAMILIES + engine.CLASSICAL)
    assert all(r["base_comparison_status"] == "AVAILABLE" for r in candidates)
    assert all(r["rms"]**2 == pytest.approx(r["mean_error"]**2 + r["spatial_rms"]**2, abs=1e-15) for r in candidates)
    assert all("signed_mean_error" in r and "base_spatial_rms" in r for r in candidates)
    assert (tmp_path / "comparisons.json").exists()
    assert len(json.loads((tmp_path / "target_coverage.json").read_text())["rows"]) == 80
    engine._verify_manifest(tmp_path, protocol, "neural_manifest.json")


def test_training_failure_remains_a_declared_outcome(prepared, tmp_path, monkeypatch):
    protocol, source, _ = prepared
    original = engine._train
    def fail(family, seed, *args, **kwargs):
        if family == "premix":
            return None, {"family": family, "seed": seed, "status": "NUMERICAL_FAILURE", "steps": 0,
                          "selection": "NO_ELIGIBLE_CHECKPOINT", "error": "Injected training failure"}
        return original(family, seed, *args, **kwargs)
    monkeypatch.setattr(engine, "_train", fail)
    summary = engine.run(protocol, source, tmp_path)
    candidates = json.loads((tmp_path / "candidates.json").read_text())["rows"]
    failed = [r for r in candidates if r["family"] == "premix"]
    assert len(failed) == 24 and all(r["status"] == "TRAINING_FAILED" for r in failed)
    assert summary["counts"]["numerical_training_failures"] == 1
    assert summary["counts"]["candidates"] == 240


def test_frozen_required_consistency_failure_prevents_diagnostics_and_seal(prepared, tmp_path, monkeypatch):
    protocol, source, _ = prepared
    original = engine._trained_consistency
    def broken(protocol, models, records, path, device, budget):
        model = models[protocol["seeds"][0], "fno_gated"]
        forward = model.forward
        model.forward = lambda state, h, equation, geometry: forward(state, h, equation, geometry) + .001
        return original(protocol, models, records, path, device, budget)
    monkeypatch.setattr(engine, "_trained_consistency", broken)
    monkeypatch.setattr(engine, "_evaluate", lambda *args: pytest.fail("Failed exact limits cannot open fresh diagnostics"))
    with pytest.raises(RuntimeError, match="exact-limit"):
        engine.run(protocol, source, tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "FAILED"
    checks = json.loads((tmp_path / "consistency.json").read_text())["rows"]
    assert checks[-1]["status"] == "FAIL" and checks[-1]["required"] is True
    assert json.loads((tmp_path / "candidates.json").read_text())["rows"] == []
    assert (tmp_path / "checkpoint_freeze.json").exists()
    assert not (tmp_path / "neural_manifest.json").exists()


def test_interruption_preserves_partial_state_and_withholds_completion(prepared, tmp_path):
    protocol, source, _ = prepared
    with pytest.raises(InterruptedError):
        engine.run(protocol, source, tmp_path, stop=SimpleNamespace(requested=True, signal_number=10))
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "INTERRUPTED"
    assert json.loads((tmp_path / "candidates.json").read_text())["rows"] == []
    assert not (tmp_path / "checkpoint_freeze.json").exists()
    assert not (tmp_path / "neural_manifest.json").exists()


def test_prepare_interruption_never_seals(tmp_path):
    with pytest.raises(InterruptedError):
        engine.prepare(engine.build_protocol("smoke"), tmp_path, stop=SimpleNamespace(requested=True, signal_number=10))
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "INTERRUPTED"
    assert not (tmp_path / "dataset_manifest.json").exists()


def test_cuda_request_cannot_fall_back(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    protocol, source, _ = prepared
    with pytest.raises(RuntimeError, match="fallback"):
        engine.run(protocol, source, tmp_path, device="cuda")
    assert not (tmp_path / "summary.json").exists()


def test_frontier_coverage_excludes_failed_uncertain_and_over_memory():
    protocol = engine.build_protocol("smoke")
    base = {"parent_id": "p", "regime": "mixed", "category": "typical", "distribution": "fresh", "grid": [8, 8],
            "seed": 1, "family": "premix", "steps": 1, "status": "COMPLETED", "upper_rms": 1e-5,
            "upper_max": 3e-4, "timing": {"wall_seconds_median": .2}, "memory_ok": True}
    rows = [base, {**base, "steps": 2, "upper_rms": 1e-7, "memory_ok": False},
            {**base, "steps": 4, "status": "TRAINING_FAILED", "upper_rms": 1e-7}]
    frontiers = engine._frontiers(rows, protocol)
    rms = {row["target"]: row for row in frontiers if row["norm"] == "rms"}
    assert rms[2e-4]["selected_steps"] == 1
    assert rms[2e-6]["status"] == "NO_FEASIBLE_CANDIDATE"


def test_regression_ratios_use_matching_parent_and_steps():
    base = dict(parent_id="p", family="strang", steps=2, status="COMPLETED", rms=.1,
                max_error=.2, mean_error=.0, spatial_rms=.1)
    rows = [base, {**base, "family": "premix", "rms": .2, "max_error": .1, "spatial_rms": .2},
            {**base, "family": "fno", "steps": 1}]
    engine._base_regression(rows)
    assert rows[1]["rms_vs_base"] == 2
    assert rows[1]["max_vs_base"] == .5
    assert rows[1]["mean_vs_base"] is None
    assert rows[2]["base_comparison_status"] == "BASE_UNAVAILABLE"
