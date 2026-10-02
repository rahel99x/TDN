"""The desktop deliverable contains source packages and excludes machine state."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_desktop", ROOT / "scripts/package_desktop.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


@pytest.mark.parametrize("name", [".venv/Scripts/python.exe", ".runtime/tmp/file.txt",
                                 "data/train.npy", "runs/run/checkpoints/last.pt", "scripts/user.env.sh",
                                 "requirements/environment-versions.json", "dist/archive.zip"])
def test_archive_excludes_machine_state(name):
    assert package.excluded(Path(name))


@pytest.mark.parametrize("name", ["tdn/data/dataset.py", "reference/temporal_core.py", "scripts/windows/Setup.ps1"])
def test_archive_preserves_all_source_packages(name):
    assert not package.excluded(Path(name))


def test_archive_refuses_symlink_source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("not source")
    try:
        (root / "source.py").symlink_to(outside)
    except OSError:
        pytest.skip("Windows account lacks symlink permission")
    with pytest.raises(ValueError, match="symlink"):
        package.source_files(root)
