"""Audit evidence must expose failed constraints and preserve completed runs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from tdn.analysis.consistency import audits
from tdn.analysis.consistency.protocol import build_protocol


def _read(path):
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    path = tmp_path_factory.mktemp("consistency-full-audit")
    protocol = build_protocol("full")
    before = deepcopy(protocol)
    rng = torch.random.get_rng_state().clone()
    summary = audits.audit(protocol, path)
    assert protocol == before
    assert torch.equal(rng, torch.random.get_rng_state())
    return path, protocol, summary


def test_full_audit_matches_frozen_coverage_and_exposes_unconstrained_controls(completed):
    path, protocol, summary = completed
    rows = _read(path / "checks.json")["rows"]
    assert summary["status"] == "COMPLETED"
    assert summary["coverage"] == {"expected": 128, "reported": 128}
    assert summary["outcomes"] == {"PASS": 110, "OBSERVED": 18}
    assert summary["correctness_failures"] == 0
    assert summary["training_performed"] is False
    assert len({row["check_id"] for row in rows}) == len(rows)
    assert {row["check_id"] for row in rows} == set(protocol["audit_case_ids"])
    assert {row["check_id"] for row in rows if row["required"]} == set(protocol["audit_required_case_ids"])
    for row in rows:
        assert row["status"] == ("PASS" if row["required"] else "OBSERVED")
        if row["required"]:
            assert row["maximum_absolute_error"] <= row["absolute_tolerance"]
    controls = [row for row in rows if not row["required"]]
    assert {row["family"] for row in controls} == {"premix", "fno", "precompress"}
    assert all(row["maximum_absolute_error"] > 0 for row in controls)
    # A zero-head initialization could trivially pass these limits. The audit
    # records nonzero randomized weights for every model/dtype instead.
    assert all(row["parameter_l2_norm"] > 0 for row in rows if "parameter_l2_norm" in row)


def test_audit_manifest_hashes_exact_canonical_artifacts_and_preserves_run(completed):
    path, protocol, _ = completed
    manifest = _read(path / "audit_manifest.json")
    assert manifest["protocol_sha256"] == audits._digest(protocol)
    assert set(manifest["artifacts"]) == {"protocol.json", "checks.json", "audit.json", "summary.json"}
    for name, fingerprint in manifest["artifacts"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == fingerprint
    snapshot = {name: (path / name).read_bytes() for name in manifest["artifacts"]}
    with pytest.raises(FileExistsError, match="Preserve"):
        audits.audit(protocol, path)
    assert all((path / name).read_bytes() == contents for name, contents in snapshot.items())


def test_mean_and_commutator_evidence_records_independent_targets_and_orders(completed):
    path, _, _ = completed
    rows = _read(path / "checks.json")["rows"]
    moment = [row for row in rows if row["check_id"].endswith("/final_mean_target")]
    assert len(moment) == 4
    for row in moment:
        assert len(row["target_mean"]) == len(row["final_mean"]) == 2
        assert row["target_mean"] == pytest.approx(row["final_mean"], abs=row["absolute_tolerance"])
        assert "not exact mean ODE" in row["claim"]
    orders = [row for row in rows if row["check_id"].endswith("small_amplitude_commutator_order")]
    assert len(orders) == 2
    for row in orders:
        assert row["observed_orders"] == pytest.approx([2., 2.], abs=row["absolute_tolerance"])
    telescoping = [row for row in rows if row["check_id"].endswith("periodic_telescoping")]
    assert len(telescoping) == 4
    assert all("mean(C)=2*kappa*r" in row["algebraic_identity"] for row in telescoping)
    for row in telescoping:
        assert "m-m^2-Var[u]" in row["mean_law_identity"]
        assert row["mean_law_error"] <= row["absolute_tolerance"]
        assert row["diffusion_mean_error"] <= row["absolute_tolerance"]
        expected = [row["reaction_rate"] * (mean - mean**2 - variance)
                    for mean, variance in zip(row["mean_state"], row["variance"])]
        assert row["physical_mean_derivative"] == pytest.approx(expected, abs=row["absolute_tolerance"])
        # These nontrivial logistic states gain mass through reaction. A mean
        # conservation constraint would fail this physical PDE identity.
        assert all(derivative > 0 for derivative in row["physical_mean_derivative"])


def test_failed_required_check_is_saved_and_cannot_be_sealed(tmp_path, monkeypatch):
    protocol = build_protocol("smoke")
    bad = audits._row(protocol["audit_case_ids"][0], 1., 1e-12)
    monkeypatch.setattr(audits, "_cases", lambda *_: iter([bad]))
    with pytest.raises(RuntimeError, match="structural audit failed"):
        audits.audit(protocol, tmp_path)
    assert _read(tmp_path / "checks.json")["rows"] == [bad]
    summary = _read(tmp_path / "summary.json")
    assert summary["status"] == "FAILED"
    assert summary["correctness_failures"] == 1
    assert summary["coverage"] == {"expected": 128, "reported": 1}
    assert not (tmp_path / "audit_manifest.json").exists()


@pytest.mark.parametrize("failure", ("incomplete", "duplicate", "relabel_required"))
def test_audit_cannot_seal_incomplete_or_relabeled_coverage(tmp_path, monkeypatch, failure):
    protocol = build_protocol("smoke")
    rows = [audits._row(identity, 0., 1e-12,
             required=identity in protocol["audit_required_case_ids"])
            for identity in protocol["audit_case_ids"]]
    if failure == "incomplete":
        rows.pop()
    elif failure == "duplicate":
        rows[-1] = deepcopy(rows[0])
    else:
        rows[0]["required"] = False
    monkeypatch.setattr(audits, "_cases", lambda *_: iter(rows))
    with pytest.raises(RuntimeError, match="frozen check plan"):
        audits.audit(protocol, tmp_path)
    assert _read(tmp_path / "summary.json")["status"] == "FAILED"
    assert not (tmp_path / "audit_manifest.json").exists()


def test_interrupt_preserves_partial_case_evidence_without_a_seal(tmp_path):
    protocol = build_protocol("smoke")
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls > 3

    with pytest.raises(InterruptedError):
        audits.audit(protocol, tmp_path, stop=stop)
    summary = _read(tmp_path / "summary.json")
    assert summary["status"] == "INTERRUPTED"
    assert summary["coverage"] == {"expected": 128, "reported": 3}
    assert len(_read(tmp_path / "checks.json")["rows"]) == 3
    assert not (tmp_path / "audit_manifest.json").exists()


def test_worker_signal_flag_interrupts_before_work_without_a_seal(tmp_path):
    stop = SimpleNamespace(requested=True, signal_number=10)
    with pytest.raises(InterruptedError, match="signal 10"):
        audits.audit(build_protocol("smoke"), tmp_path, stop=stop)
    summary = _read(tmp_path / "summary.json")
    assert summary["status"] == "INTERRUPTED"
    assert summary["coverage"] == {"expected": 128, "reported": 0}
    assert _read(tmp_path / "checks.json")["rows"] == []
    assert not (tmp_path / "audit_manifest.json").exists()


def test_budget_failure_is_visible_without_a_seal(tmp_path, monkeypatch):
    times = iter((0., 1201.))
    monkeypatch.setattr(audits, "time", SimpleNamespace(monotonic=lambda: next(times)))
    with pytest.raises(TimeoutError, match="wall-time budget"):
        audits.audit(build_protocol("smoke"), tmp_path)
    summary = _read(tmp_path / "summary.json")
    assert summary["status"] == "INTERRUPTED"
    assert summary["coverage"] == {"expected": 128, "reported": 0}
    assert not (tmp_path / "audit_manifest.json").exists()


def test_mutated_protocol_is_rejected_before_evidence_is_written(tmp_path):
    protocol = build_protocol("smoke")
    protocol["audit_seconds"] = 2000
    with pytest.raises(ValueError, match="immutable declared plan"):
        audits.audit(protocol, tmp_path)
    assert not (tmp_path / "protocol.json").exists()
