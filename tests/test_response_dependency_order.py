"""Exercise serialized prerequisite verification through the real stage CLI.

These are CPU infrastructure tests, not native CUDA or scientific validation.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from tdn.analysis.response.protocol import ancestors, build_protocol


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("response_dependency_cli", ROOT / "scripts/response.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)
TARGETS = ("fd_same_grid", "galerkin_same_grid", "galerkin_projected_fine")


@pytest.mark.parametrize("kind", ["evaluate", "cost"])
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("invalid_lineage", [False, True])
def test_serialized_smoke_verifies_selected_trials_before_consumers(tmp_path, monkeypatch, capsys, kind, target, invalid_lineage):
    import tdn.analysis.response.engine as engine
    import tdn.reporting as reporting
    import tdn.runtime.metadata as metadata
    import tdn.runtime.preflight as preflight
    import tdn.runtime.storage as storage

    # Keep the native three-target DAG, but explicitly run only mocked CPU
    # orchestration.  sorted-key JSON is the native coordinator's wire format.
    protocol = build_protocol("smoke", targets=list(TARGETS))
    protocol["execution_scope"] = "CPU_IMPLEMENTATION_SMOKE"
    protocol = json.loads(json.dumps(protocol, sort_keys=True))
    stage = f"{kind}-{target}"
    required = ancestors(protocol, stage)
    trial_stages = [name for name in required if protocol["units"][name]["kind"] == "train"]
    assert len(trial_stages) > 1
    assert list(protocol["units"]).index("selection") < min(list(protocol["units"]).index(name) for name in trial_stages)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True))
    software = {"python": "test-cpu", "source_tree_sha256": "fixture-source"}
    paths = {}
    bindings = {}
    for name in required:
        path = tmp_path / name
        path.mkdir()
        (path / "workflow-seal.json").write_text(json.dumps({"fixture_stage": name}))
        paths[name] = path
        bindings[name] = {"run_dir": str(path), "workflow_seal_sha256": cli.file_digest(path / "workflow-seal.json")}

    verified = []
    def verify_execution(path, frozen, *, source_tree_sha256):
        name = Path(path).name
        assert frozen == protocol and source_tree_sha256 == software["source_tree_sha256"]
        verified.append(name)
        prior = {parent: bindings[parent] for parent in ancestors(protocol, name)}
        if invalid_lineage and name == "selection":
            prior[trial_stages[0]] = {**prior[trial_stages[0]], "workflow_seal_sha256": "different-trial"}
        return {"execution": {"stage": name, "execution_mode": "local-cpu", "software": software,
                "prerequisites": prior}}

    dispatched = []
    def run_stage(frozen, name, path, *, prerequisites, **kwargs):
        assert frozen == protocol and name == stage
        assert prerequisites == paths
        dispatched.append(name)
        raise RuntimeError("TEST_STOP_AFTER_PREREQUISITE_VERIFICATION")

    for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "TDN_EXECUTION_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(preflight, "execution_mode", lambda: "local-cpu")
    monkeypatch.setattr(preflight, "verify_runtime", lambda *args: None)
    monkeypatch.setattr(storage, "configure_storage", lambda: None)
    monkeypatch.setattr(metadata, "software_metadata", lambda: software)
    monkeypatch.setattr(cli, "verify_execution", verify_execution)
    monkeypatch.setattr(engine, "run_stage", run_stage)
    monkeypatch.setattr(reporting, "attach_report", lambda *args, **kwargs: (tmp_path / "tower", False))
    monkeypatch.setattr(reporting, "emit", lambda *args, **kwargs: None)
    argv = ["--stage", stage, "--protocol", str(protocol_path), "--run-dir", str(tmp_path / "consumer"),
            "--local-root", str(ROOT), "--device", "cpu"]
    for name, path in paths.items():
        argv += ["--prerequisite-dir", f"{name}={path}"]
    assert cli.main(argv) == 1
    stderr = capsys.readouterr().err
    if invalid_lineage:
        assert not dispatched
        assert "Prerequisites belong to different frozen lineages" in stderr
        assert not (tmp_path / "consumer").exists()
    else:
        assert dispatched == [stage]
        assert verified == required
        assert "TEST_STOP_AFTER_PREREQUISITE_VERIFICATION" in stderr
