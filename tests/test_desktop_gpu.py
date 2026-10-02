"""Native desktop CUDA readiness, without Slurm, Bash, or a compiler backend.

The launcher selects this file explicitly and sets TDN_REQUIRE_GPU_TESTS=1,
so a machine without a working CUDA installation fails before collection.
These are numerical and replay checks, rather than a performance claim.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from reference.temporal_core import psi3, temporal_defect, temporal_jets
from tdn.config import load_config
from tdn.data import generate_dataset
from tdn.models import build_model
from tdn.numerics import Equation, Geometry, split_step, validate_state
from tdn.numerics.reference import choose_substeps, refined_reference
from tdn.runtime.precision import reference_precision
from tdn.train import load_checkpoint, train


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.environ.get("TDN_EXECUTION_MODE") != "desktop",
                       reason="Select desktop execution mode for native GPU readiness"),
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="No CUDA device; desktop GPU coverage remains unvalidated"),
]


@pytest.fixture(autouse=True)
def eager_ieee_policy():
    reference_precision()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(23)
    torch.cuda.manual_seed_all(23)


def test_desktop_fp64_temporal_values_jets_and_gradients():
    x = torch.cat([torch.zeros(1), torch.logspace(-12, 6, 221)]).double()
    torch.testing.assert_close(psi3(x.cuda()).cpu(), psi3(x), atol=1e-14, rtol=3e-13)
    amplitudes = torch.randn(8, 1, 4, dtype=torch.float64, requires_grad=True)
    rates = torch.rand_like(amplitudes, requires_grad=True)
    h = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    cpu_output = temporal_defect(amplitudes, rates, h)
    cpu_gradients = torch.autograd.grad(cpu_output.square().sum(), (amplitudes, rates, h))
    gpu_values = [value.detach().cuda().requires_grad_() for value in (amplitudes, rates, h)]
    gpu_output = temporal_defect(*gpu_values)
    gpu_gradients = torch.autograd.grad(gpu_output.square().sum(), gpu_values)
    torch.testing.assert_close(gpu_output.cpu(), cpu_output, atol=1e-13, rtol=1e-12)
    for actual, expected in zip(gpu_gradients, cpu_gradients):
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-12, rtol=1e-11)
    for value in (0., .04, .5, 2.):
        step = torch.tensor(value, device="cuda", dtype=torch.float64, requires_grad=True)
        jets = temporal_jets(gpu_values[0], gpu_values[1], step)
        derivative = jets[0].sum()
        for expected in jets[1:]:
            derivative, = torch.autograd.grad(derivative, step, create_graph=True)
            torch.testing.assert_close(derivative, expected.sum(), atol=2e-11, rtol=2e-10)


def test_desktop_fp64_teacher_and_eager_fp32_split_invariants():
    geometry = Geometry((6, 6), (1., 1.))
    equation = Equation(.013, 2.)
    state = .2 + .6 * torch.rand(1, 1, 6, 6, dtype=torch.float64)
    h = .08
    substeps = max(16, choose_substeps(h, equation, geometry))
    teacher_cpu = refined_reference(state, h, equation, geometry, substeps)
    teacher_gpu = refined_reference(state.cuda(), h, equation, geometry, substeps)
    assert teacher_cpu.accepted and teacher_gpu.accepted
    assert teacher_cpu.refinement_substeps == teacher_gpu.refinement_substeps
    torch.testing.assert_close(teacher_gpu.state.cpu(), teacher_cpu.state,
                               atol=2e-12, rtol=2e-11)
    split_cpu = split_step(state, h, equation, geometry)
    split_gpu = split_step(state.cuda(), h, equation, geometry)
    torch.testing.assert_close(split_gpu.cpu(), split_cpu, atol=2e-12, rtol=2e-11)
    split_fp32 = split_step(state.float().cuda(), h, equation, geometry)
    assert split_fp32.dtype == torch.float32
    validate_state(teacher_gpu.state)
    validate_state(split_fp32)
    torch.testing.assert_close(split_fp32.double().cpu(), split_cpu, atol=5e-7, rtol=3e-6)


def test_desktop_eager_fp32_temporal_model_output_and_gradient_parity():
    cpu_model = build_model(8, {"family": "temporal_mlp", "width": 16, "modes": 4})
    with torch.no_grad():
        cpu_model.amplitude.weight.normal_(0., .04)
        cpu_model.amplitude.bias.normal_(0., .04)
        cpu_model.rate.weight.normal_(0., .02)
    gpu_model = copy.deepcopy(cpu_model).cuda()
    cpu_features = torch.randn(13, 8, requires_grad=True)
    cpu_step = torch.tensor(.2, requires_grad=True)
    gpu_features = cpu_features.detach().cuda().requires_grad_()
    gpu_step = cpu_step.detach().cuda().requires_grad_()
    cpu_output = cpu_model(cpu_features, cpu_step)
    gpu_output = gpu_model(gpu_features, gpu_step)
    assert gpu_output.dtype == torch.float32
    torch.testing.assert_close(gpu_output.cpu(), cpu_output, atol=2e-8, rtol=3e-5)
    # Nonzero amplitude/rate heads ensure this measures real decoder and
    # parameter gradients rather than comparing two zero-initialized outputs.
    cpu_output.square().sum().backward()
    gpu_output.square().sum().backward()
    torch.testing.assert_close(gpu_features.grad.cpu(), cpu_features.grad, atol=2e-9, rtol=5e-5)
    torch.testing.assert_close(gpu_step.grad.cpu(), cpu_step.grad, atol=2e-9, rtol=5e-5)
    assert torch.count_nonzero(gpu_model.rate.weight.grad) > 0
    for cpu_parameter, gpu_parameter in zip(cpu_model.parameters(), gpu_model.parameters()):
        assert cpu_parameter.grad is not None and gpu_parameter.grad is not None
        assert torch.isfinite(gpu_parameter.grad).all()
        torch.testing.assert_close(gpu_parameter.grad.cpu(), cpu_parameter.grad,
                                   atol=2e-9, rtol=5e-5)


def _assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual.cpu(), expected.cpu())
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected):
            _assert_tree_equal(item, reference)
    else:
        assert actual == expected


def test_desktop_cuda_checkpoint_resume_matches_uninterrupted(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    config["problem"]["grid"] = [4, 4]
    config["data"].update(train_count=2, validation_count=1, diagnostic_count=1)
    config["training"].update(max_steps=6, checkpoint_every_steps=2,
                               validation_every_steps=3, microbatch_cells=16)
    config["runtime"].update(soft_vram_gib=8, soft_vram_fraction=.6)
    data = tmp_path / "data"
    generate_dataset(config, data)
    whole = tmp_path / "whole"
    replay = tmp_path / "replay"
    uninterrupted = train(config, data, whole, device="cuda")
    paused = train(config, data, replay, device="cuda", max_steps=2)
    assert paused["status"] == "PAUSED_BUDGET"
    resumed = train(config, data, replay, device="cuda", resume=replay / "checkpoints" / "last.pt")
    assert uninterrupted["status"] == resumed["status"] == "COMPLETE"
    assert uninterrupted["history"] == resumed["history"]
    assert uninterrupted["validation"] == resumed["validation"]
    expected = load_checkpoint(whole)
    actual = load_checkpoint(replay)
    for key in ("model_state", "optimizer_state", "scheduler_state", "sampler"):
        _assert_tree_equal(actual[key], expected[key])
    _assert_tree_equal(actual["rng_state"], expected["rng_state"])
    assert actual["sampler"]["committed_cursor"] == actual["global_step"] == 6
    assert (replay / "checkpoints" / "last.previous.pt").is_file()
