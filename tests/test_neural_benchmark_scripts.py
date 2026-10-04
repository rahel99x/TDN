"""Thin wrapper argument contracts; no scheduler or numerical work is invoked."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Bash wrappers require a POSIX host")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def wrappers(tmp_path):
    root = tmp_path / "project with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("carc_neural_benchmarks.sh", "neural_benchmarks_local.sh"):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    for name in ("carc_research.sh", "research_local.sh"):
        (scripts / name).write_text(
            "#!/usr/bin/env bash\nexec " + str(sys.executable) +
            " -c 'import json,sys; print(json.dumps(sys.argv[1:])); sys.exit(37)' \"$@\"\n")
    return root


def invoke(root, name, *args):
    return subprocess.run(["bash", str(root / "scripts" / name), *args], cwd=root,
                          text=True, capture_output=True, timeout=10, check=False)


def forwarded(root, name, *args):
    result = invoke(root, name, *args)
    assert result.returncode == 37, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("args", [(), ("--submit",), ("start", "--smoke", "--run-id", "quoted run")])
def test_carc_start_defaults_to_neural_config_and_preserves_arguments(wrappers, args):
    expected = list(args[1:] if args and args[0] == "start" else args)
    assert forwarded(wrappers, "carc_neural_benchmarks.sh", *args) == [
        "start", "--neural-benchmarks", "--config", str(wrappers / "scripts" / ".." / "configs" / "neural-benchmarks.yaml"),
        *expected]


@pytest.mark.parametrize("args", [
    ("status", "latest"), ("logs", "run-123", "--lines", "100"),
    ("collect", "run-123"), ("benchmark", "cpu-123", "--submit"),
])
def test_readonly_and_benchmark_commands_do_not_inject_a_config(wrappers, args):
    assert forwarded(wrappers, "carc_neural_benchmarks.sh", *args) == list(args)


@pytest.mark.parametrize("option", [("--config", "configs/custom with spaces.yaml"),
                                    ("--config=configs/custom with spaces.yaml",)])
def test_carc_explicit_config_overrides_default(wrappers, option):
    assert forwarded(wrappers, "carc_neural_benchmarks.sh", *option, "--submit") == [
        "start", "--neural-benchmarks", *option, "--submit"]


def test_carc_explicit_neural_marker_is_not_duplicated(wrappers):
    args = forwarded(wrappers, "carc_neural_benchmarks.sh", "--neural-benchmarks", "--submit")
    assert args.count("--neural-benchmarks") == 1


def test_local_defaults_to_neural_config_and_preserves_path_quoting(wrappers):
    assert forwarded(wrappers, "neural_benchmarks_local.sh", "--run-dir", "runs/with spaces", "--smoke") == [
        "--config", str(wrappers / "scripts" / ".." / "configs" / "neural-benchmarks.yaml"),
        "--run-dir", "runs/with spaces", "--smoke"]


def test_local_explicit_config_is_forwarded_unchanged(wrappers):
    args = ["--config", "configs/custom with spaces.yaml", "--smoke"]
    assert forwarded(wrappers, "neural_benchmarks_local.sh", *args) == args


@pytest.mark.parametrize("name", ["carc_neural_benchmarks.sh", "neural_benchmarks_local.sh"])
def test_wrapper_help_never_dispatches_to_a_worker(wrappers, name):
    result = invoke(wrappers, name, "--help")
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout and "neural-benchmarks.yaml" in result.stdout
    syntax = subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)],
                            text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr
