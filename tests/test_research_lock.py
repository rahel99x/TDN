"""Project venv lock contracts using local POSIX locks, without Slurm or NFS."""
from __future__ import annotations

import errno
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

fcntl = pytest.importorskip("fcntl", reason="CARC venv locking requires POSIX")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def controller(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "research_lock_controller_test", ROOT / "scripts" / "research_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project with spaces"
    root.mkdir()
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module.cw, "ROOT", root)
    return module


def lock_attempt(path, *, shared=False):
    """A separate process probes the same lock file using real flock semantics."""
    program = """
import fcntl, sys
with open(sys.argv[1], 'a+b') as handle:
    try:
        kind = fcntl.LOCK_SH if sys.argv[2] == 'shared' else fcntl.LOCK_EX
        fcntl.flock(handle.fileno(), kind | fcntl.LOCK_NB)
    except BlockingIOError:
        print('BLOCKED')
    else:
        print('ACQUIRED')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(path), "shared" if shared else "exclusive"],
        capture_output=True, text=True, timeout=5, check=True)
    return result.stdout.strip()


def record_open_handles(monkeypatch):
    opened = []
    real_open = Path.open

    def tracked_open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        if path.name == "carc-phase.lock":
            opened.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", tracked_open)
    return opened


def test_shared_lock_descriptor_supports_nfs_style_read_lock(controller):
    with controller.acquire_venv_lock() as handle:
        assert handle.readable() and handle.writable()
        assert fcntl.fcntl(handle.fileno(), fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDWR
        # Linux NFS can translate flock into a whole-file POSIX lock. A shared
        # POSIX lock on a write-only descriptor raises EBADF even on local disk.
        fcntl.lockf(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)


def test_concurrent_readers_block_exclusive_writer_until_all_close(controller):
    first = controller.acquire_venv_lock()
    second = None
    path = controller.ROOT / ".cache" / "carc-phase.lock"
    try:
        second = controller.acquire_venv_lock()
        assert lock_attempt(path, shared=True) == "ACQUIRED"
        assert lock_attempt(path) == "BLOCKED"
        first.close()
        assert lock_attempt(path) == "BLOCKED"
    finally:
        first.close()
        if second is not None:
            second.close()
    assert lock_attempt(path) == "ACQUIRED"


def test_exclusive_workflow_contention_closes_shared_handle(controller, monkeypatch):
    path = controller.ROOT / ".cache" / "carc-phase.lock"
    path.parent.mkdir()
    opened = record_open_handles(monkeypatch)
    with path.open("a+b") as installer:
        fcntl.flock(installer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="exclusive") as error:
            controller.acquire_venv_lock()
        assert isinstance(error.value.__cause__, BlockingIOError)
        assert len(opened) == 2 and opened[-1].closed


def test_lock_system_error_is_preserved_and_never_called_contention(controller, monkeypatch):
    opened = record_open_handles(monkeypatch)
    failure = OSError(errno.EBADF, "Bad file descriptor")

    def reject_lock(*args):
        raise failure

    monkeypatch.setattr(fcntl, "flock", reject_lock)
    with pytest.raises(OSError, match="Cannot acquire project venv shared lock") as error:
        controller.acquire_venv_lock()
    assert error.value.__cause__ is failure
    assert "Bad file descriptor" in str(error.value)
    assert "exclusive" not in str(error.value)
    assert len(opened) == 1 and opened[0].closed


def test_portable_eacces_contention_is_recognized(controller, monkeypatch):
    opened = record_open_handles(monkeypatch)
    failure = PermissionError(errno.EACCES, "Lock unavailable")

    def reject_lock(*args):
        raise failure

    monkeypatch.setattr(fcntl, "flock", reject_lock)
    with pytest.raises(ValueError, match="exclusively") as error:
        controller.acquire_venv_lock()
    assert error.value.__cause__ is failure
    assert len(opened) == 1 and opened[0].closed
