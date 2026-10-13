"""Exact actual-CUDA case inventory for this research namespace.

This module declares identities; collecting or passing a CPU-skipped suite
does not establish native readiness. The workflow checks the entire JUnit
multiset and rejects every skipped, failed, missing or duplicated case.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


GPU_TEST_FILES = (
    'tests/test_tangent_response_gpu.py',
    'tests/test_tangent_response_training_gpu.py',
)
GPU_TEST_SOURCE_FILES = GPU_TEST_FILES + (
    'tests/test_tangent_response_numerics.py',
    'tests/test_tangent_response_models.py',
    'tests/test_tangent_response_training.py',
)

_TARGETS = ('fd_same_grid', 'galerkin_same_grid')
_NUMERICAL_MODULE = 'tests.test_tangent_response_gpu'
_TRAINING_MODULE = 'tests.test_tangent_response_training_gpu'


def _case(module, name):
    # Dicts survive a protocol JSON roundtrip with exactly the same identity.
    return dict(classname=module, name=name)


NUMERICAL_GPU_CASES = tuple([
    _case(_NUMERICAL_MODULE, f'test_tangent_cuda_operator_odd_even_parity[{size}-{target}]')
    for size in (5, 6) for target in _TARGETS
] + [
    _case(_NUMERICAL_MODULE, f'{name}[{target}]')
    for name in (
        'test_tangent_cuda_homogeneous_action_parity',
        'test_tangent_cuda_forced_full_error_parity',
        'test_tangent_cuda_physics_time_gradients',
        'test_tangent_cuda_physical_nyquist_norm',
        'test_tangent_cuda_nulls_identity_and_mean_coupling',
        'test_tangent_cuda_same_path_composition',
    ) for target in _TARGETS
] + [_case(_NUMERICAL_MODULE, 'test_tangent_cuda_actual_allocated_device')])

TRAINING_GPU_CASES = tuple([
    _case(_TRAINING_MODULE, f'{name}[{target}]')
    for name in (
        'test_tangent_cuda_model_head_and_factor_gradients',
        'test_tangent_cuda_signed_causality_nulls_and_mean',
    ) for target in _TARGETS
] + [
    _case(_TRAINING_MODULE, f'{name}[{family}]')
    for name in (
        'test_tangent_cuda_cache_identity',
        'test_tangent_cuda_bridge_and_complete_cached_parity',
        'test_tangent_cuda_exact_staged_resume',
        'test_tangent_cuda_fresh_process_selected_checkpoint_reload',
    ) for family in ('learned_response', 'residual_fno')
])
GPU_TEST_CASES = NUMERICAL_GPU_CASES + TRAINING_GPU_CASES


def test_source_hashes(root=None):
    """Hash mandatory test sources for the frozen protocol, never a mock run."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    result = {}
    for relative in GPU_TEST_SOURCE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Mandatory Tangent Response CUDA source missing: {relative}')
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
