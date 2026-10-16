"""Independent small-matrix checks; no retained-bank fits or new references."""
import json

import numpy as np
import pytest
from scipy.linalg import lstsq
import torch

from tdn.analysis.response_preconditioner.fitting import fit_parent_balanced_ridge
from tdn.analysis.response_preconditioner.regularization_fitting import (
    fit_parent_balanced_projection, fit_parent_balanced_tikhonov,
)


def sample():
    rng = np.random.default_rng(20261017)
    A, b = rng.normal(size=(19, 4)), rng.normal(size=19)
    parents = ["a"]*3 + ["b"]*7 + ["c"]*9
    q = np.linspace(.3, 1.4, len(b))
    L = np.array([[1., .3, 0, 0], [0, 2., -.4, .1], [.2, 0, .5, 1.7], [0, 0, .1, -.2], [.4, 0, 0, 0]])
    w0 = np.array([.4, -.8, .12, 1.1])
    return A, b, parents, q, L, w0


def weighted(A, b, parents, q):
    omega = np.empty(len(b))
    for p in set(parents):
        mask = np.array([x == p for x in parents])
        omega[mask] = q[mask]/q[mask].sum()/len(set(parents))
    return A*np.sqrt(omega[:, None]), b*np.sqrt(omega), omega


def test_general_penalty_and_prior_match_independent_unscaled_scipy_lstsq():
    A, b, parents, q, L, w0 = sample()
    lam = .07
    result = fit_parent_balanced_tikhonov(A, b, parents, ridge=lam, row_weights=q, penalty=L, prior=w0)
    Aw, bw, _ = weighted(A, b, parents, q)
    B = np.vstack((Aw, np.sqrt(lam)*L))
    rhs = np.concatenate((bw, np.sqrt(lam)*L@w0))
    expected = lstsq(B, rhs, lapack_driver="gelsd")[0]
    actual = result["coefficients"].numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
    d = result["diagnostics"]
    assert d["objective"] == pytest.approx(np.sum((Aw@actual-bw)**2)+lam*np.sum((L@(actual-w0))**2), rel=3e-15)
    assert d["eligible"] and d["reasons"] == []
    assert d["original_penalty_preserved"] and d["prior_rhs_preserved"]
    assert not d["singular_value_truncation"] and not d["certificate"]
    json.dumps(d, allow_nan=False)


def test_gradient_and_curvature_use_the_full_generalized_hessian_not_lambda():
    A = np.array([[1., 0.], [0., .02], [.1, .01]])
    b, L, w0, lam = np.array([1., 2., -.3]), np.diag([2., .01]), np.array([.3, -.7]), .8
    result = fit_parent_balanced_tikhonov(A, b, ["a"]*3, ridge=lam, penalty=L, prior=w0)
    w = result["coefficients"].numpy()
    Aw, bw = A/np.sqrt(3.), b/np.sqrt(3.)
    Hnormal = Aw.T@Aw + lam*L.T@L
    gradient = 2*(Aw.T@(Aw@w-bw)+lam*L.T@L@(w-w0))
    mu = np.linalg.eigvalsh(Hnormal)[0]
    d = result["diagnostics"]
    assert mu < lam/1000
    assert d["minimum_normal_eigenvalue"] == pytest.approx(mu, rel=3e-13)
    assert d["minimum_hessian_eigenvalue"] == pytest.approx(2*mu, rel=3e-13)
    assert d["normal_residual_norm"] == pytest.approx(d["gradient_norm"]/2, rel=2e-15)
    assert d["optimality_gap_upper_bound"] == pytest.approx(d["gradient_norm"]**2/(4*mu), rel=3e-13)
    assert np.linalg.norm(gradient) < 1e-12
    assert d["eligible"]


@pytest.mark.parametrize("ridge", [1e-10, .2])
def test_identity_penalty_matches_original_ridge_fitter(ridge):
    A, b, parents, q, _, _ = sample()
    old = fit_parent_balanced_ridge(A, b, parents, ridge=ridge, row_weights=q)
    new = fit_parent_balanced_tikhonov(A, b, parents, ridge=ridge, row_weights=q)
    torch.testing.assert_close(new["coefficients"], old["coefficients"], atol=3e-14, rtol=3e-13)
    for key in ("data_loss", "ridge_loss", "objective", "zero_objective"):
        assert new["diagnostics"][key] == pytest.approx(old["diagnostics"][key], rel=2e-14)
    assert new["diagnostics"]["eligible"]


def test_generalized_effective_dof_matches_explicit_small_hat_matrix():
    A, b, parents, q, L, w0 = sample()
    result = fit_parent_balanced_tikhonov(A, b, parents, ridge=.07, penalty=L, prior=w0, row_weights=q)
    Aw, _, _ = weighted(A, b, parents, q)
    hat = Aw@np.linalg.solve(Aw.T@Aw+.07*L.T@L, Aw.T)
    assert result["diagnostics"]["generalized_effective_degrees_of_freedom"] == pytest.approx(np.trace(hat), rel=2e-14)


def test_identity_effective_dof_matches_singular_shrinkage_formula():
    A, b, parents, q, _, _ = sample()
    result = fit_parent_balanced_tikhonov(A, b, parents, ridge=.4, row_weights=q)
    Aw, _, _ = weighted(A, b, parents, q)
    s = np.linalg.svd(Aw, compute_uv=False)
    assert result["diagnostics"]["generalized_effective_degrees_of_freedom"] == pytest.approx(np.sum(s*s/(s*s+.4)), rel=2e-14)


def test_reparameterization_transports_penalty_and_prior_without_changing_predictions():
    A, b, parents, q, L, w0 = sample()
    transform = np.array([[.01, .3, 0, 0], [0, 2., .2, 0], [0, 0, 30., .4], [.01, 0, 0, 5.]])
    original = fit_parent_balanced_tikhonov(A, b, parents, ridge=.03, penalty=L, prior=w0, row_weights=q)
    changed = fit_parent_balanced_tikhonov(A@transform, b, parents, ridge=.03,
        penalty=L@transform, prior=np.linalg.solve(transform,w0), row_weights=q)
    w = original["coefficients"].numpy()
    transformed = transform@changed["coefficients"].numpy()
    np.testing.assert_allclose(transformed, w, rtol=5e-11, atol=2e-13)
    np.testing.assert_allclose(A@transformed, A@w, rtol=2e-10, atol=2e-13)
    assert original["diagnostics"]["objective"] == pytest.approx(changed["diagnostics"]["objective"], rel=3e-14)
    assert original["diagnostics"]["generalized_effective_degrees_of_freedom"] == pytest.approx(
        changed["diagnostics"]["generalized_effective_degrees_of_freedom"], rel=2e-13)
    assert changed["diagnostics"]["eligible"]


def test_rank_deficient_data_and_semidefinite_penalty_can_identify_jointly():
    result = fit_parent_balanced_tikhonov([[1., 0]], [2.], ["a"], ridge=.3,
                                        penalty=[[0., 2.]], prior=[4., 3.])
    torch.testing.assert_close(result["coefficients"], torch.tensor([2.,3.],dtype=torch.float64), atol=2e-15, rtol=2e-15)
    d = result["diagnostics"]
    assert d["original_design_numerical_rank"] == d["penalty_rank"] == 1
    assert d["penalty_nullity"] == 1 and d["original_augmented_numerical_rank"] == 2
    assert d["eligible"] and not d["singular_value_truncation"]


def test_common_data_and_penalty_nullspace_is_ineligible_and_has_no_gap():
    result = fit_parent_balanced_tikhonov([[1., 0],[2.,0]], [1., 3.], ["a","a"], ridge=.3,
                                        penalty=[[2., 0.]], prior=[0., 8.])
    d = result["diagnostics"]
    assert not d["eligible"] and d["diagnostic_only"]
    assert d["status"] == "NON_IDENTIFIABLE_DIAGNOSTIC"
    assert d["optimality_gap_upper_bound"] is None and d["minimum_normal_eigenvalue"] is None
    assert d["generalized_effective_degrees_of_freedom"] is None
    assert d["zero_objective_columns"] == 1
    assert d["singular_value_truncation"] and not d["optimality_verified"]
    assert result["coefficients"][1] == 0
    assert "ORIGINAL_AUGMENTED_SYSTEM_NUMERICALLY_RANK_DEFICIENT" in d["reasons"]
    json.dumps(d, allow_nan=False)


def test_zero_penalty_is_identified_by_full_rank_data():
    A, b, parents, q, _, w0 = sample()
    result = fit_parent_balanced_tikhonov(A, b, parents, ridge=.3, penalty=np.zeros((2,4)), prior=w0, row_weights=q)
    Aw, bw, _ = weighted(A,b,parents,q)
    np.testing.assert_allclose(result["coefficients"].numpy(), lstsq(Aw,bw)[0], rtol=4e-13, atol=2e-14)
    assert result["diagnostics"]["penalty_rank"] == 0
    assert result["diagnostics"]["eligible"]


def test_zero_data_and_identity_penalty_recover_nonzero_prior():
    w0 = [.3, -.7]
    result = fit_parent_balanced_tikhonov(np.zeros((3,2)), [0.,0.,0.], ["a","b","b"], ridge=.2, prior=w0)
    np.testing.assert_allclose(result["coefficients"].numpy(), w0, atol=2e-15, rtol=2e-15)
    assert result["diagnostics"]["eligible"]
    assert result["diagnostics"]["generalized_effective_degrees_of_freedom"] == 0


def test_complete_parent_duplication_preserves_equal_parent_fit():
    A,b,parents,q,L,w0=sample()
    a=fit_parent_balanced_tikhonov(A,b,parents,ridge=.2,penalty=L,prior=w0,row_weights=q)
    duplicated=fit_parent_balanced_tikhonov(np.vstack((A,A[:3])),np.concatenate((b,b[:3])),
        parents+parents[:3],ridge=.2,penalty=L,prior=w0,row_weights=np.concatenate((q,q[:3])))
    torch.testing.assert_close(a["coefficients"],duplicated["coefficients"],atol=2e-14,rtol=2e-13)


def test_list_inputs_preserve_float64_precision():
    A,b=[[1.000000000001],[1.000000000003]],[2.000000000007,2.000000000011]
    result=fit_parent_balanced_tikhonov(A,b,["a","a"],ridge=.13,penalty=[[1.000000000005]],prior=[.100000000003])
    a=np.array(A).ravel(); target=np.array(b); L=1.000000000005; prior=.100000000003
    expected=(np.mean(a*target)+.13*L*L*prior)/(np.mean(a*a)+.13*L*L)
    assert float(result["coefficients"][0]) == pytest.approx(expected, abs=8e-16)


@pytest.mark.parametrize("rank_deficient", [False,True])
def test_unregularized_projection_matches_independent_scipy_but_never_deploys(rank_deficient):
    A,b,parents,q,_,_=sample()
    if rank_deficient:
        A[:,2]=2*A[:,0]; A[:,3]=0
    result=fit_parent_balanced_projection(A,b,parents,row_weights=q)
    Aw,bw,_=weighted(A,b,parents,q)
    expected=lstsq(Aw,bw,cond=np.finfo(float).eps*max(Aw.shape),lapack_driver="gelsd")[0]
    np.testing.assert_allclose(result["coefficients"].numpy(),expected,rtol=4e-13,atol=2e-14)
    d=result["diagnostics"]
    assert not d["eligible"] and not d["deployable"] and d["diagnostic_only"]
    assert d["optimality_gap_upper_bound"] is None
    assert d["singular_value_truncation"] == rank_deficient
    assert d["data_loss"] == pytest.approx(np.sum((Aw@expected-bw)**2),rel=2e-14)
    json.dumps(d,allow_nan=False)


def test_projection_zero_design_and_target_has_finite_zero_minimum_norm():
    result=fit_parent_balanced_projection(np.zeros((2,3)),[0.,0.],["a","b"])
    assert torch.equal(result["coefficients"],torch.zeros(3,dtype=torch.float64))
    assert result["diagnostics"]["original_design_numerical_rank"] == 0
    json.dumps(result["diagnostics"],allow_nan=False)


def test_diagonal_general_penalty_has_the_known_nonzero_prior_solution():
    a=np.array([.2,1.7,3.]); l=np.array([2.,.03,.7]); b=np.array([1.,-.8,.4]); w0=np.array([.5,-2.,.3])
    lam=.08
    result=fit_parent_balanced_tikhonov(np.diag(a),b,["a"]*3,ridge=lam,penalty=np.diag(l),prior=w0)
    expected=(a*b/3+lam*l*l*w0)/(a*a/3+lam*l*l)
    np.testing.assert_allclose(result["coefficients"].numpy(),expected,rtol=2e-14,atol=2e-14)
    assert result["diagnostics"]["eligible"]


def test_gap_evaluation_avoids_intermediate_gradient_square_overflow():
    # Every input, objective and final gap is finite even though ||grad||²
    # would overflow if evaluated before division by the curvature.
    result=fit_parent_balanced_tikhonov([[1e150]],[1e150],["a"],ridge=1.,penalty=[[1e150]],prior=[.3])
    d=result["diagnostics"]
    assert d["eligible"] and d["optimality_gap_upper_bound"] is not None
    assert d["optimality_gap_upper_bound"] <= d["optimality_tolerance"]
    json.dumps(d,allow_nan=False)


def test_unrepresentable_squared_curvature_is_ineligible_but_json_finite():
    result=fit_parent_balanced_tikhonov([[1e154]],[1e154],["a"],ridge=1.,penalty=[[1e154]],prior=[.3])
    d=result["diagnostics"]
    assert not d["eligible"] and d["minimum_normal_eigenvalue"] is None
    assert d["minimum_hessian_eigenvalue"] is None and d["optimality_gap_upper_bound"] is None
    assert "NO_FINITE_POSITIVE_NUMERICAL_CURVATURE" in d["reasons"]
    json.dumps(d,allow_nan=False)


@pytest.mark.parametrize("bad", ["ridge_zero","ridge_nan","ridge_bool","negative_ridge","complex_design",
    "nan_design","target_shape","complex_target","weight_zero","weight_nan","weight_shape",
    "parents_count","empty_parent","penalty_shape","penalty_empty","penalty_nan","penalty_complex",
    "prior_shape","prior_nan","prior_complex","tolerance_zero","tolerance_nan"])
def test_malformed_generalized_inputs_rejected(bad):
    A,b,parents,q,L,w0=sample(); ridge=.1; tol=1e-8
    if bad=="ridge_zero": ridge=0
    if bad=="ridge_nan": ridge=float("nan")
    if bad=="ridge_bool": ridge=True
    if bad=="negative_ridge": ridge=-1
    if bad=="complex_design": A=A.astype(complex); A[0,0]+=1j
    if bad=="nan_design": A[0,0]=np.nan
    if bad=="target_shape": b=b[:,None]
    if bad=="complex_target": b=b.astype(complex); b[0]+=1j
    if bad=="weight_zero": q[0]=0
    if bad=="weight_nan": q[0]=np.nan
    if bad=="weight_shape": q=q[:-1]
    if bad=="parents_count": parents=parents[:-1]
    if bad=="empty_parent": parents[0]=""
    if bad=="penalty_shape": L=L[:,:-1]
    if bad=="penalty_empty": L=L[:0]
    if bad=="penalty_nan": L[0,0]=np.nan
    if bad=="penalty_complex": L=L.astype(complex); L[0,0]+=1j
    if bad=="prior_shape": w0=w0[:-1]
    if bad=="prior_nan": w0[0]=np.nan
    if bad=="prior_complex": w0=w0.astype(complex); w0[0]+=1j
    if bad=="tolerance_zero": tol=0
    if bad=="tolerance_nan": tol=np.nan
    with pytest.raises(ValueError):
        fit_parent_balanced_tikhonov(A,b,parents,ridge=ridge,penalty=L,prior=w0,row_weights=q,optimality_rtol=tol)


@pytest.mark.parametrize("bad", ["nan","parents","zero_weight","target_shape"])
def test_projection_rejects_bad_physical_inputs(bad):
    A,b,parents,q,_,_=sample()
    if bad=="nan": b[0]=np.nan
    if bad=="parents": parents=parents[:-1]
    if bad=="zero_weight": q[0]=0
    if bad=="target_shape": b=b[:,None]
    with pytest.raises(ValueError):
        fit_parent_balanced_projection(A,b,parents,row_weights=q)
