"""Portable file publication and source archives without Git installed."""
from types import SimpleNamespace

import pytest
import torch

from tdn.data import provenance
from tdn.runtime import metadata
from tdn.train import checkpoint


def test_windows_directory_flush_avoids_posix_directory_descriptors(monkeypatch, tmp_path):
    def forbidden(*_args):
        pytest.fail("Windows must not open a directory using the POSIX fsync path")
    monkeypatch.setattr(provenance, "os", SimpleNamespace(name="nt", open=forbidden))
    provenance.fsync_directory(tmp_path)


def test_atomic_checkpoint_retains_backup_without_hardlink_support(monkeypatch, tmp_path):
    def unavailable(*_args):
        raise OSError("Filesystem does not support hard links")
    current, previous = tmp_path / "last.pt", tmp_path / "last.previous.pt"
    checkpoint._atomic_torch_save(current, {"generation": 1})
    monkeypatch.setattr(checkpoint.os, "link", unavailable)
    checkpoint._atomic_torch_save(current, {"generation": 2}, previous)
    assert torch.load(current, weights_only=True)["generation"] == 2
    assert torch.load(previous, weights_only=True)["generation"] == 1
    assert not list(tmp_path.glob("*.pending"))


def test_source_provenance_does_not_require_git(monkeypatch):
    monkeypatch.setattr(provenance.shutil, "which", lambda _name: None)
    report = provenance.source_provenance()
    assert report["commit"] == "unversioned"
    assert report["files"] and len(report["python_source_hash"]) == 64
    assert any(name.startswith("tdn/numerics/") for name in report["files"])
    assert all("\\" not in name for name in report["files"])


def test_software_metadata_does_not_require_git(monkeypatch):
    monkeypatch.setattr(metadata.shutil, "which", lambda _name: None)
    report = metadata.software_metadata()
    assert report["git_commit"] is None
    assert report["git_available"] is False
    assert report["tracked_or_untracked_changes"] is None
    assert len(report["source_tree_sha256"]) == 64
