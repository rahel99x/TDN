"""Independent optimization and weighting checks for the affine stage."""
import json
import numpy as np
import pytest
import torch

from tdn.analysis.response_preconditioner.fitting import fit_parent_balanced_ridge


def sample():
    generator = torch.Generator().manual_seed(194)
    design = torch.randn(17, 4, generator=generator, dtype=torch.float64)
    truth = torch.tensor([.2, -.7, 1.3, .04], dtype=torch.float64)
    target = design @ truth
    parents = ["a"]*3 + ["b"]*5 + ["c"]*9
    q = torch.linspace(.3, 1.7, 17, dtype=torch.float64)
    return design, target, parents, q, truth


def weights(parents, q):
    result = torch.empty_like(q)
    for parent in set(parents):
        mask = torch.tensor([x == parent for x in parents])
        result[mask] = q[mask]/q[mask].sum()/len(set(parents))
    return result


def test_matches_independent_unscaled_numpy_svd_and_objective():
    A, b, parents, q, _ = sample()
    ridge = .03
    result = fit_parent_balanced_ridge(A, b, parents, ridge=ridge, row_weights=q)
    omega = weights(parents, q)
    weighted_A = A * omega.sqrt()[:, None]
    weighted_b = b * omega.sqrt()
    augmented = np.concatenate((weighted_A.numpy(), np.sqrt(ridge)*np.eye(A.shape[1])))
    rhs = np.concatenate((weighted_b.numpy(), np.zeros(A.shape[1])))
    U, S, Vh = np.linalg.svd(augmented, full_matrices=False)
    expected = Vh.T @ ((U.T @ rhs)/S)
    np.testing.assert_allclose(result["coefficients"].numpy(), expected, rtol=2e-13, atol=2e-14)
    w = result["coefficients"]
    objective = (omega*(A@w-b).square()).sum() + ridge*w.square().sum()
    diag = result["diagnostics"]
    assert diag["objective"] == pytest.approx(float(objective), rel=2e-15)
    assert diag["optimality_verified"] and diag["baseline_nonregression_verified"]
    assert diag["objective"] < diag["zero_objective"]
    assert diag["certificate"] is False
    json.dumps(diag, allow_nan=False)


def test_preserves_original_ridge_under_large_column_scaling():
    A, b, parents, q, _ = sample()
    A *= torch.tensor([1e-3, 1., 1e2, 1e4], dtype=torch.float64)
    ridge = .2
    result = fit_parent_balanced_ridge(A, b, parents, ridge=ridge, row_weights=q)
    omega = weights(parents, q)
    augmented = torch.cat((A*omega.sqrt()[:, None], np.sqrt(ridge)*torch.eye(4, dtype=torch.float64)))
    rhs = torch.cat((b*omega.sqrt(), torch.zeros(4, dtype=torch.float64)))
    expected = torch.linalg.lstsq(augmented, rhs, driver="gels").solution
    torch.testing.assert_close(result["coefficients"], expected, atol=2e-13, rtol=3e-10)
    assert result["diagnostics"]["original_penalty_preserved"]
    assert not result["diagnostics"]["singular_value_truncation"]


@pytest.mark.parametrize("ridge", (1e-10, .03))
def test_tiny_and_zero_columns_use_augmented_norm_without_changing_objective(ridge):
    A, b, parents, q, _ = sample()
    # A source-time feature can be tiny without being exactly zero. Its data
    # norm alone would magnify the ridge penalty by approximately 1e14.
    tiny = 1e-14 * (A[:, 0] + A[:, 1]).square()
    A = torch.cat((A, tiny[:, None], torch.zeros_like(tiny[:, None])), dim=1)
    result = fit_parent_balanced_ridge(A, b, parents, ridge=ridge, row_weights=q)
    omega = weights(parents, q)
    weighted_A = (A * omega.sqrt()[:, None]).numpy()
    weighted_b = (b * omega.sqrt()).numpy()

    # Independently solve in the original coefficient coordinates, once by
    # augmented SVD and once by the direct ridge filter on the data SVD.
    augmented = np.concatenate((weighted_A, np.sqrt(ridge)*np.eye(A.shape[1])))
    rhs = np.concatenate((weighted_b, np.zeros(A.shape[1])))
    U, singular, Vh = np.linalg.svd(augmented, full_matrices=False)
    unscaled = Vh.T @ ((U.T @ rhs)/singular)
    U, singular, Vh = np.linalg.svd(weighted_A, full_matrices=False)
    direct = Vh.T @ ((singular/(singular**2 + ridge)) * (U.T @ weighted_b))
    actual = result["coefficients"].numpy()
    for expected in (unscaled, direct):
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)
        np.testing.assert_allclose(A.numpy() @ actual, A.numpy() @ expected,
                                   rtol=3e-12, atol=3e-13)
        expected_objective = np.sum((weighted_A @ expected-weighted_b)**2) + ridge*np.sum(expected**2)
        assert result["diagnostics"]["objective"] == pytest.approx(expected_objective, rel=3e-12)

    diag = result["diagnostics"]
    expected_scales = np.hypot(np.linalg.norm(weighted_A, axis=0), np.sqrt(ridge))
    np.testing.assert_allclose(diag["column_scales"], expected_scales, rtol=3e-15)
    assert diag["column_scaling_strategy"] == "original_augmented_column_l2_norm"
    assert diag["zero_training_columns"] == 1
    assert diag["minimum_column_scale"] == pytest.approx(np.sqrt(ridge))
    assert 0 < diag["minimum_scaled_ridge_penalty"] <= diag["maximum_scaled_ridge_penalty"]
    assert diag["maximum_scaled_ridge_penalty"] == pytest.approx(1., rel=3e-15)
    assert diag["optimality_verified"] and diag["original_penalty_preserved"]
    assert diag["original_design_numerical_rank"] < A.shape[1]
    assert len(diag["augmented_singular_values"]) == A.shape[1]
    assert not diag["singular_value_truncation"]

    old_scales = np.linalg.norm(weighted_A, axis=0)
    old_scales[old_scales == 0] = 1
    old_augmented = augmented / old_scales
    assert diag["augmented_condition_number"] < np.linalg.cond(old_augmented) / 1e6
    json.dumps(diag, allow_nan=False)


def test_known_affine_coefficients_recovered_as_ridge_tends_to_zero():
    A, b, parents, q, truth = sample()
    result = fit_parent_balanced_ridge(A, b, parents, ridge=1e-10, row_weights=q)
    torch.testing.assert_close(result["coefficients"], truth, atol=8e-10, rtol=8e-10)


def test_python_list_inputs_preserve_float64_precision_before_fitting():
    A = [[1.000000000001], [1.000000000003]]
    b = [2.000000000007, 2.000000000011]
    q = [1.000000000001, 1.000000000005]
    result = fit_parent_balanced_ridge(A, b, ["same", "same"], ridge=.13, row_weights=q)
    weights64 = torch.tensor(q, dtype=torch.float64)
    weights64 /= weights64.sum()
    a64, b64 = torch.tensor(A, dtype=torch.float64).flatten(), torch.tensor(b, dtype=torch.float64)
    expected = (weights64*a64*b64).sum() / ((weights64*a64.square()).sum() + .13)
    torch.testing.assert_close(result["coefficients"][0], expected, atol=8e-16, rtol=0)
    rounded = fit_parent_balanced_ridge(torch.tensor(A, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32), ["same", "same"], ridge=.13,
        row_weights=torch.tensor(q, dtype=torch.float32))
    assert abs(float(result["coefficients"][0] - rounded["coefficients"][0])) > 1e-12


def test_equal_parent_weight_invariant_to_complete_parent_row_duplication():
    A, b, parents, q, _ = sample()
    result = fit_parent_balanced_ridge(A, b, parents, ridge=.1, row_weights=q)
    duplicated = fit_parent_balanced_ridge(torch.cat((A, A[:3])), torch.cat((b,b[:3])),
        parents+parents[:3], ridge=.1, row_weights=torch.cat((q,q[:3])))
    torch.testing.assert_close(result["coefficients"], duplicated["coefficients"], atol=2e-14, rtol=2e-13)


def test_rank_deficient_and_zero_columns_are_not_truncated():
    A, b, parents, q, _ = sample()
    A[:,1] = A[:,0]
    A[:,3] = 0.
    result = fit_parent_balanced_ridge(A, b, parents, ridge=.1, row_weights=q)
    coeff = result["coefficients"]
    assert abs(float(coeff[3])) < 1e-14
    assert float(coeff[0]) == pytest.approx(float(coeff[1]), abs=1e-14)
    diag = result["diagnostics"]
    assert diag["original_design_numerical_rank"] == 2
    assert diag["original_design_condition_number"] is None
    assert diag["optimality_verified"]
    assert len(diag["augmented_singular_values"]) == 4
    json.dumps(diag, allow_nan=False)


def test_zero_target_has_unique_zero_solution():
    A, b, parents, _, _ = sample()
    result = fit_parent_balanced_ridge(A, b*0, parents, ridge=.1)
    torch.testing.assert_close(result["coefficients"], torch.zeros(4,dtype=torch.float64), atol=0, rtol=0)
    assert result["diagnostics"]["optimality_gap_upper_bound"] == 0.


@pytest.mark.parametrize("bad", ("ridge_zero", "ridge_nan", "weight_zero", "weight_nan", "nan", "parents", "shape"))
def test_invalid_fit_inputs_rejected(bad):
    A, b, parents, q, _ = sample()
    ridge = .1
    if bad == "ridge_zero": ridge = 0.
    if bad == "ridge_nan": ridge = float("nan")
    if bad == "weight_zero": q[0] = 0.
    if bad == "weight_nan": q[0] = float("nan")
    if bad == "nan": A[0,0] = float("nan")
    if bad == "parents": parents = parents[:-1]
    if bad == "shape": b = b[:,None]
    with pytest.raises(ValueError):
        fit_parent_balanced_ridge(A, b, parents, ridge=ridge, row_weights=q)
