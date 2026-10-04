"""Neural suite provenance and independent, bounded GPU prerequisites."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from test_research_carc import controller, finish_cpu, plan
from tdn.numerics import Equation, Geometry, split_step
from tdn.research import build_research_model
from tdn.research.protocol import DEFAULT, NEURAL_FAMILIES, benchmark_suite, digest, validate_config

ROOT = Path(__file__).resolve().parents[1]


def finish_neural(controller, *, baseline_trained=True, tdn_trained=True):
    workflow = plan(controller, "neural-source", "--neural-benchmarks")
    base = finish_cpu(controller, workflow, headroom=False)
    protocol_path = base / "experiment/protocol.json"
    protocol = controller.cw.read_json(protocol_path)
    protocol["benchmark_suite"] = "neural-benchmarks"
    controller.cw.atomic_json(protocol_path, protocol)
    summary_path = base / "experiment/summary.json"
    summary = controller.cw.read_json(summary_path)
    summary["training"] = [
        {"family": "reaction_clock", "benchmark_role": "tdn", "status": "COMPLETED",
         "selected_step": 8 if tdn_trained else 0, "selected_parameters_changed": tdn_trained},
        {"family": "fno_split", "benchmark_role": "neural_baseline", "status": "COMPLETED",
         "selected_step": 8 if baseline_trained else 0, "selected_parameters_changed": baseline_trained},
    ]
    controller.cw.atomic_json(summary_path, summary)
    manifest_path = base / "experiment/manifest.json"
    manifest = controller.cw.read_json(manifest_path)
    manifest["protocol_sha256"] = digest(protocol)
    for name in ("protocol.json", "summary.json"):
        manifest["files"][name] = controller.cw.digest(base / "experiment" / name)
    controller.cw.atomic_json(manifest_path, manifest)
    (base / "experiment/COMPLETED").write_text(controller.cw.digest(manifest_path))
    return workflow, base


def test_neural_config_keeps_every_control_and_all_scientific_limits():
    config = yaml.safe_load((ROOT / "configs/neural-benchmarks.yaml").read_text())
    assert validate_config(config) == config
    assert config["families"] == list(NEURAL_FAMILIES)
    assert benchmark_suite(config) == "neural-benchmarks"
    for key in ("tolerance", "train_horizons", "heldout_horizons", "rollout_time", "benchmark_steps", "rollout_weight"):
        assert config[key] == DEFAULT[key]
    missing = copy.deepcopy(config)
    missing["families"].remove("fno")
    with pytest.raises(ValueError, match="Retain every"):
        validate_config(missing)


def test_neural_gpu_preview_uses_completed_trained_pairs_without_classical_headroom(controller, monkeypatch, capsys):
    workflow, _ = finish_neural(controller)
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("Preview called scheduler"))
    assert controller.main(["benchmark", workflow["run_id"], "--run-id", "neural-gpu"]) == 0
    output = capsys.readouterr().out
    assert "Suite: neural-benchmarks" in output
    assert "--gpus-per-task=a100:1" in output and "--time=00:30:00" in output
    gpu = controller.prepare(controller.parser().parse_args(["benchmark", workflow["run_id"]]))
    assert gpu["benchmark_suite"] == "neural-benchmarks"


@pytest.mark.parametrize("tdn,baseline", [(False, True), (True, False), (False, False)])
def test_neural_gpu_rejects_untrained_initializations(controller, tdn, baseline):
    workflow, _ = finish_neural(controller, baseline_trained=baseline, tdn_trained=tdn)
    with pytest.raises(ValueError, match="trained validation-selected"):
        controller.completed_source(controller.workflow_path(workflow["run_id"]))


def test_neural_suite_cannot_relabel_an_old_classical_workflow(controller):
    workflow, base = finish_neural(controller)
    workflow["benchmark_suite"] = "architecture"
    controller.cw.atomic_json(base / "research-workflow.json", workflow)
    with pytest.raises(ValueError, match="suite differs"):
        controller.completed_source(base / "research-workflow.json")


def test_worker_collects_real_cpu_junit_and_requires_immutable_suite(controller):
    workflow = plan(controller, "neural-worker", "--neural-benchmarks")
    commands = controller.worker_commands(workflow)
    assert "--junitxml" in commands[0][1]
    assert commands[1][1][commands[1][1].index("--expected-suite") + 1] == "neural-benchmarks"


def test_generic_mlp_is_a_genuine_h_conditioned_control_with_same_physical_base():
    temporal = build_research_model("temporal_mlp").double()
    generic = build_research_model("generic_mlp").double()
    assert temporal.model.is_h_independent and not generic.model.is_h_independent
    assert generic.model.base.body[0].in_features == temporal.model.base.body[0].in_features + 1
    state = (.2 + .6 * torch.rand(2, 1, 8, 8, dtype=torch.float64)).requires_grad_()
    h = torch.tensor([.02, .05], dtype=torch.float64, requires_grad=True)
    equation, geometry = Equation(.01, 2.), Geometry((8, 8), (1., 1.))
    torch.testing.assert_close(generic(state, h, equation, geometry), split_step(state, h, equation, geometry), atol=0., rtol=0.)
    with torch.no_grad():
        generic.model.base.amplitude.weight.normal_(0., .01)
    result = generic(state, h, equation, geometry)
    gradients = torch.autograd.grad(result.square().mean(), (state, h, *generic.parameters()))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert gradients[1].abs().sum() > 0


def test_engine_rejects_suite_mismatch_before_creating_science_output(tmp_path):
    run_dir = tmp_path / "wrong-suite"
    result = subprocess.run([sys.executable, "scripts/research.py", "run", "--config", "configs/neural-benchmarks.yaml",
                             "--expected-suite", "architecture", "--run-dir", str(run_dir)],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "Configuration suite differs" in result.stderr
    assert not run_dir.exists()
