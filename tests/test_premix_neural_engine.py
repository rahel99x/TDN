"""Meaningful isolation, integrity, actual-training and outcome regressions."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.premix import neural


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    path = tmp_path_factory.mktemp("premix-neural-data")
    protocol = neural.build_protocol("smoke")
    result = neural.prepare(protocol, path)
    yield protocol, path, result
    torch.set_num_threads(original_threads)


def test_full_plan_is_frozen_disjoint_and_stresses_resolution_spectrum_and_horizon():
    protocol = neural.build_protocol("full")
    neural.validate_protocol(protocol)
    assert [sum(parent["split"] == split for parent in protocol["parents"])
            for split in ("train", "validation", "diagnostic")] == [48, 24, 48]
    assert len({parent["seed"] for parent in protocol["parents"]}) == 120
    diagnostics = [parent for parent in protocol["parents"] if parent["split"] == "diagnostic"]
    assert {parent["category"] for parent in diagnostics} == {"favorable", "typical", "adverse"}
    assert {parent["regime"] for parent in diagnostics} == {row[0] for row in neural.REGIMES}
    assert all(parent["grid"] == [64, 64] for parent in diagnostics if parent["regime"] == "grid_transfer")
    assert all(parent["horizons"] == [.96] for parent in diagnostics if parent["regime"] == "long_rollout")
    assert protocol["seeds"] == [840011, 840021, 840031]
    assert protocol["updates"] == 300
    with pytest.raises(ValueError, match="immutable"):
        neural.validate_protocol({**protocol, "updates": 301})
    with pytest.raises(ValueError):
        neural.validate_protocol({**protocol, "updates": 300.0})
    with pytest.raises(ValueError):
        neural.build_protocol("expanded")


def test_high_high_initial_pair_has_no_low_input_but_has_low_product():
    parent = next(parent for parent in neural.build_protocol("full")["parents"] if parent["regime"] == "high_pair")
    initial = neural._initial(parent)
    fluctuation = initial - initial.mean()
    spectrum = torch.fft.fftn(fluctuation, dim=(-2, -1), norm="forward")
    product = torch.fft.fftn(fluctuation.square(), dim=(-2, -1), norm="forward")
    assert abs(spectrum[0, 0, 1, 1]) < 1e-14
    assert abs(product[0, 0, 1, 1]) > .005


def test_prepare_accepted_fp64_references_and_train_only_normalization(prepared):
    protocol, path, summary = prepared
    assert summary["status"] == "COMPLETED"
    assert summary["counts"] == {"parents": 20, "references": 32, "accepted_references": 32}
    dataset, normalization = neural._load_dataset(protocol, path)
    assert all(parent["initial"].dtype == torch.float64 for parent in dataset["parents"])
    assert all(reference["accepted"] for parent in dataset["parents"] for reference in parent["references"].values())
    assert len({parent["initial_state_sha256"] for parent in dataset["parents"]}) == 20
    record = json.loads((path / "normalization.json").read_text())
    train = [parent for parent in dataset["parents"] if parent["split"] == "train"]
    expected = neural.fit_feature_normalization(((parent["initial"], neural.Equation(parent["kappa"], parent["reaction_rate"]),
        neural.Geometry(tuple(parent["grid"]), (1., 1.))) for parent in train), t_ref=protocol["t_ref"])
    assert record["parent_ids"] == [parent["parent_id"] for parent in train]
    assert torch.equal(normalization[0], expected[0].float())
    assert torch.equal(normalization[1], expected[1].float())
    assert not (path / "dataset.partial.pt").exists()
    with pytest.raises(FileExistsError, match="fresh"):
        neural.prepare(protocol, path)


def test_tampered_dataset_is_rejected_before_torch_load(prepared, tmp_path, monkeypatch):
    protocol, source, _ = prepared
    import shutil
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    with (tmp_path / "dataset.pt").open("ab") as handle:
        handle.write(b"tampered")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Do not deserialize an unverified artifact"))
    with pytest.raises(ValueError, match="integrity"):
        neural._load_dataset(protocol, tmp_path)


def test_validation_normalization_cannot_be_relabelled_as_training(prepared, tmp_path):
    protocol, source, _ = prepared
    import shutil
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    normalization = json.loads((tmp_path / "normalization.json").read_text())
    normalization["parent_ids"] = [parent["parent_id"] for parent in protocol["parents"] if parent["split"] == "validation"]
    neural.write_json(tmp_path / "normalization.json", normalization)
    names = list(json.loads((tmp_path / "dataset_manifest.json").read_text())["artifacts"])
    neural._manifest(tmp_path, names, protocol, "dataset_manifest.json")
    with pytest.raises(ValueError, match="training-only"):
        neural._load_dataset(protocol, tmp_path)


def test_actual_training_freezes_every_family_before_any_diagnostic(prepared, tmp_path, monkeypatch):
    protocol, source, _ = prepared
    original = neural._evaluate
    inspections = []

    def audited(protocol, dataset, models, records, path, device, budget, candidates):
        freeze = json.loads((path / "checkpoint_freeze.json").read_text())
        assert freeze["all_training_completed_before_diagnostics"] is True
        assert len(records) == len(protocol["seeds"]) * len(neural.FAMILIES)
        assert all(record["status"] == "COMPLETED" and record["steps"] == 2 for record in records)
        assert len(freeze["artifacts"]) == 5
        for name, fingerprint in freeze["artifacts"].items():
            assert neural.file_digest(path / name) == fingerprint
        inspections.append(True)
        return original(protocol, dataset, models, records, path, device, budget, candidates)

    monkeypatch.setattr(neural, "_evaluate", audited)
    summary = neural.run(protocol, source, tmp_path)
    assert inspections == [True]
    assert summary["actual_neural_training"] is True
    assert summary["actual_training_device"] == "cpu"
    assert summary["fnopaper_reproduction"] is False
    records = json.loads((tmp_path / "training.json").read_text())["rows"]
    assert len({row["sample_schedule_sha256"] for row in records}) == 1
    assert all(row["diagnostics_seen_during_training"] is False for row in records)
    assert all(row["training_seconds"] > 0 and row["parameter_count"] > 0 for row in records)
    for row in records:
        assert [point["step"] for point in row["history"]] == [0, 1, 2]
        assert row["selected_step"] == min(row["history"], key=lambda point: point["validation_loss"])["step"]
        if row["selected_step"] == 0:
            assert row["selection"] == "SELECTED_INITIALIZATION"
            assert not row["selected_parameters_changed"]
    candidates = json.loads((tmp_path / "candidates.json").read_text())["rows"]
    assert len(candidates) == 192
    assert {row["family"] for row in candidates} == set(neural.FAMILIES + neural.CLASSICAL)
    assert {tuple(row["grid"]) for row in candidates} == {(8, 8), (16, 16)}
    assert all(row["admissible"] for row in candidates)
    assert all(row["upper_rms"] >= row["rms"] for row in candidates)
    assert all(row["upper_max"] >= row["max_error"] for row in candidates)
    assert all(len(row["timing"]["wall_seconds_raw"]) == 1 for row in candidates)
    assert all(row["timing"]["device"] == "cpu" for row in candidates)
    assert len(json.loads((tmp_path / "frontiers.json").read_text())["rows"]) == 768
    neural._verify_manifest(tmp_path, protocol, "neural_manifest.json")
    with pytest.raises(FileExistsError):
        neural.run(protocol, source, tmp_path)


def test_frontiers_exclude_failed_uncertain_and_over_memory_candidates():
    protocol = neural.build_protocol("smoke")
    base = dict(parent_id="p", regime="mixed", category="typical", distribution="test", grid=[8, 8],
                seed=1, family="premix", steps=1, status="COMPLETED", upper_rms=3e-5,
                upper_max=3e-4, timing={"wall_seconds_median": .2}, memory_ok=True)
    rows = [base, {**base, "steps": 2, "upper_rms": 1e-6, "upper_max": 1e-6,
                   "timing": {"wall_seconds_median": .01}, "memory_ok": False},
                  {**base, "steps": 4, "upper_rms": 1e-6, "upper_max": 1e-6,
                   "status": "INVALID_TRAJECTORY"}]
    frontiers = neural._frontiers(rows, protocol)
    rms = {row["target"]: row for row in frontiers if row["norm"] == "rms"}
    maximum = {row["target"]: row for row in frontiers if row["norm"] == "max"}
    assert rms[2e-4]["status"] == "FEASIBLE" and rms[2e-4]["selected_steps"] == 1
    assert maximum[2e-4]["status"] == "NO_FEASIBLE_CANDIDATE"
    assert rms[2e-5]["status"] == "NO_FEASIBLE_CANDIDATE"


def test_error_decomposition_and_norm_specific_uncertainty():
    target = torch.full((1, 1, 2, 2), .5, dtype=torch.float64)
    value = target + torch.tensor([[[[.1, -.1], [.1, .3]]]], dtype=torch.float64)
    record = neural._errors(value, {"state": target, "uncertainty_rms": .001,
                                    "uncertainty_max_bound": .002})
    assert record["rms"] ** 2 == pytest.approx(record["mean_error"] ** 2 + record["spatial_rms"] ** 2)
    assert record["mean_error"] == pytest.approx(.1)
    assert record["max_error"] == pytest.approx(.3)
    assert record["upper_rms"] == pytest.approx(record["rms"] + .001)
    assert record["upper_max"] == pytest.approx(.302)


def test_mock_cuda_memory_policy_accounts_for_initial_free_memory(monkeypatch):
    # This policy unit check supplies no GPU performance/readiness evidence.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (20 * 2**30, 40 * 2**30))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 2**20)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2**21)
    budget = neural._RunBudget(10, None, "cuda")
    assert budget.memory["soft_budget_bytes"] == 16 * 2**30
    budget.observe(force=True)
    assert budget.memory["observations"] == 1
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 17 * 2**30)
    with pytest.raises(MemoryError, match="budget"):
        budget.observe(force=True)


def test_stop_before_reference_generation_retains_interruption_summary(tmp_path):
    with pytest.raises(InterruptedError):
        neural.prepare(neural.build_protocol("smoke"), tmp_path,
                       stop=SimpleNamespace(requested=True, signal_number=10))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "INTERRUPTED"
    assert summary["counts"]["completed_parents"] == 0
    assert not (tmp_path / "dataset_manifest.json").exists()


def test_stop_before_training_never_opens_diagnostics(prepared, tmp_path, monkeypatch):
    protocol, source, _ = prepared
    monkeypatch.setattr(neural, "_evaluate", lambda *args: pytest.fail("Diagnostics must not open before training completes"))
    with pytest.raises(InterruptedError):
        neural.run(protocol, source, tmp_path, stop=SimpleNamespace(requested=True, signal_number=10))
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "INTERRUPTED"
    assert json.loads((tmp_path / "candidates.json").read_text())["rows"] == []
    assert not (tmp_path / "checkpoint_freeze.json").exists()


def test_missing_cuda_cannot_silently_run_on_cpu(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    protocol, source, _ = prepared
    with pytest.raises(RuntimeError, match="fallback"):
        neural.run(protocol, source, tmp_path, device="cuda")
    assert not (tmp_path / "summary.json").exists()


def test_numerical_training_failure_remains_a_planned_diagnostic_outcome(prepared, tmp_path, monkeypatch):
    protocol, source, _ = prepared
    # Inject at the training boundary while preserving the real evaluation and
    # other four actual optimizations; no failed architecture may disappear.
    original_train = neural._train

    def fail(family, seed, *args, **kwargs):
        if family == "premix":
            return None, {"family": family, "seed": seed, "status": "NUMERICAL_FAILURE", "steps": 0,
                          "selection": "NO_ELIGIBLE_CHECKPOINT", "error": "Deliberate numerical failure"}
        return original_train(family, seed, *args, **kwargs)

    monkeypatch.setattr(neural, "_train", fail)
    summary = neural.run(protocol, source, tmp_path)
    rows = json.loads((tmp_path / "candidates.json").read_text())["rows"]
    failed = [row for row in rows if row["family"] == "premix"]
    assert len(failed) == 24 and all(row["status"] == "TRAINING_FAILED" for row in failed)
    assert summary["counts"]["numerical_training_failures"] == 1
    assert summary["counts"]["candidates"] == 192
