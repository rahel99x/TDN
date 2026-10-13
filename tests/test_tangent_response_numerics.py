"""Full-field response checks independent of architecture or scientific gates."""
import numpy as np
import pytest
import torch

from tdn.analysis.response import numerics as n
from tdn.analysis.tangent_response import tangent as t
from tdn.analysis.tangent_response.oracle import NumpyEquation, independent_solve, refined_teacher

TARGETS = ('fd_same_grid', 'galerkin_same_grid')


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1); torch.manual_seed(11901001)
    yield
    torch.set_num_threads(old)


def case(size=5):
    geom = n.Geometry((size, size), (1., 1.)); eq = n.Physics(.013, 2.4)
    x = torch.arange(size, dtype=torch.float64)/size
    a = (.34+.08*torch.cos(2*torch.pi*x[:, None])+.03*torch.sin(2*torch.pi*x[None, :]))[None, None]
    z = (torch.cos(2*torch.pi*x[:, None])+.2*torch.sin(2*torch.pi*x[None, :])).expand(size, size)[None, None]
    return a, z, eq, geom


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('size', (5, 6))
def test_operator_independent_numpy_product_jvp_and_dense_tiny_verification(target, size):
    a, z, eq, geom = case(size); independent = NumpyEquation(eq, geom, target)
    actual = t.tangent_action(a, z, eq, geom, target)
    expected = independent.tangent(a.numpy()[0, 0], z.numpy()[0, 0])
    torch.testing.assert_close(actual[0, 0], torch.from_numpy(expected), atol=3e-13, rtol=3e-13)
    derivative = torch.autograd.functional.jvp(lambda u:n.rhs(u, eq, geom, target), a, z)[1]
    torch.testing.assert_close(actual, derivative, atol=3e-13, rtol=3e-13)
    fd = (n.rhs(a+1e-5*z, eq, geom, target)-n.rhs(a-1e-5*z, eq, geom, target))/(2e-5)
    torch.testing.assert_close(actual, fd, atol=2e-10, rtol=2e-10)
    matrix = independent.dense_tangent(a.numpy()[0, 0])
    np.testing.assert_allclose(matrix@z.numpy().reshape(-1), expected.reshape(-1), atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize('target', TARGETS)
def test_independent_split_path_derivative_residual_and_full_error_identity(target):
    a, z, eq, geom = case(6); independent = NumpyEquation(eq, geom, target)
    path = t.AnchorPath(a, eq, geom, target, reaction_substeps=4)
    row = path.evaluate(.06); other = independent.anchor_path(a.numpy()[0, 0], .06, reaction_substeps=4)
    for key in ('state', 'derivative', 'residual'):
        torch.testing.assert_close(row[key][0, 0], torch.from_numpy(other[key]), atol=2e-13, rtol=2e-12)
    plus = independent.anchor_path(a.numpy()[0, 0], .06001, reaction_substeps=4)['state']
    minus = independent.anchor_path(a.numpy()[0, 0], .05999, reaction_substeps=4)['state']
    np.testing.assert_allclose(other['derivative'], (plus-minus)/.00002, atol=2e-9, rtol=2e-8)
    error = .02*z+.01
    direct = n.rhs(row['state']+error, eq, geom, target)-row['derivative']
    expanded = t.tangent_action(row['state'], error, eq, geom, target)-n.scalar(eq.reaction_rate,error)*n.product(error,error,geom,target)-row['residual']
    torch.testing.assert_close(direct, expanded, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(path(0.), a, atol=0, rtol=0)


@pytest.mark.parametrize('target', TARGETS)
def test_full_field_spatial_feedback_couples_centered_probe_into_mean(target):
    a, z, eq, geom = case()
    assert abs(float(n.mean(z))) < 1e-15
    assert abs(float(n.mean(t.tangent_action(a,z,eq,geom,target)))) > .1
    assert abs(float(n.mean(t.mean_action(a,z,eq,geom,target)))) < 1e-14
    torch.testing.assert_close(t.tangent_action(a,z,eq,geom,target)-t.mean_action(a,z,eq,geom,target),
                               t.spatial_feedback(a,z,eq,geom,target), atol=3e-13, rtol=3e-13)
    path = t.AnchorPath(a, n.Physics(0., eq.reaction_rate), geom, target)
    U = t.response_action(path,path.eq,geom,target,z,0.,.04,steps=8)
    G = t.matched_mean_action(path,path.eq,geom,target,z,0.,.04,steps=8)
    assert (U-G).abs().max() > .005  # Zero diffusion is not a response null.
    assert abs(float(n.mean(U))) > 1e-4


@pytest.mark.parametrize('target', TARGETS)
def test_constant_background_analytic_response_identity_and_composition(target):
    a, z, eq, geom = case(); constant = torch.full_like(a,.23)
    path = lambda time:constant
    U = t.response_action(path,eq,geom,target,z,.01,.07,steps=16)
    expected = n.heat(z,.06,eq,geom,target)*torch.exp(torch.tensor(eq.reaction_rate*(1-2*.23)*.06,dtype=z.dtype))
    torch.testing.assert_close(U,expected,atol=5e-11,rtol=5e-11)
    torch.testing.assert_close(t.response_action(path,eq,geom,target,z,.07,.07,steps=2),z,atol=0,rtol=0)
    moving = t.AnchorPath(a,eq,geom,target)
    first = t.response_action(moving,eq,geom,target,z,.01,.04,steps=16)
    composed = t.response_action(moving,eq,geom,target,first,.04,.07,steps=16)
    direct = t.response_action(moving,eq,geom,target,z,.01,.07,steps=32)
    torch.testing.assert_close(composed,direct,atol=3e-13,rtol=3e-13)
    with pytest.raises(ValueError,match='Reversed'): t.response_action(moving,eq,geom,target,z,.04,.03)


@pytest.mark.parametrize('target', TARGETS)
def test_forced_response_and_nonlinear_controls_match_independent_full_error_oracle(target):
    a, _, eq, geom = case(); path=t.AnchorPath(a,eq,geom,target,reaction_substeps=4)
    for mode in ('mean','tangent'):
        actual=t.forced_response(path,eq,geom,target,.04,steps=32,mode=mode)
        other,metadata=independent_solve(a,.04,eq,geom,target,kind='forced',mode=mode,reaction_substeps=4,tolerance=1e-12)
        torch.testing.assert_close(actual,other,atol=2e-11,rtol=2e-7)
        assert metadata['dense'] is False
    full=t.forced_response(path,eq,geom,target,.04,steps=32,full_error=True)
    teacher,_=independent_solve(a,.04,eq,geom,target,tolerance=1e-12)
    torch.testing.assert_close(path(.04)+full,teacher,atol=2e-11,rtol=2e-11)
    p1=t.forced_response(path,eq,geom,target,.04,steps=32,nonlinear_iterations=1)
    p2=t.forced_response(path,eq,geom,target,.04,steps=32,nonlinear_iterations=2)
    assert (p2-full).norm() <= (p1-full).norm()+1e-12


def test_sampled_path_derivative_and_source_refinement_are_separate_from_integration():
    a,_,eq,geom=case();path=t.AnchorPath(a,eq,geom,'fd_same_grid')
    sampled=t.SampledAnchorPath(path,.06,nodes=5)
    row=sampled.evaluate(.025)
    diff=(sampled(.025001)-sampled(.024999))/.000002
    torch.testing.assert_close(row['derivative'],diff,atol=5e-10,rtol=5e-9)
    value,meta=t.refined_path_response(path,eq,geom,'fd_same_grid',.06,steps=16,initial_nodes=4,max_nodes=8,tolerance=1e-12)
    assert value.shape==a.shape and meta['axis']=='anchor/source interpolation nodes'
    assert meta['fixed_tangent_steps']==16 and meta['uncertainty_is_certificate'] is False


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('mode', ('mean', 'tangent'))
def test_actual_source_quadrature_matches_forced_ode_and_independent_solver(target,mode):
    a,_,eq,geom=case();path=t.AnchorPath(a,eq,geom,target,reaction_substeps=4)
    actual=t.source_quadrature(path,eq,geom,target,.04,nodes=8,steps=32,mode=mode)
    forced=t.forced_response(path,eq,geom,target,.04,steps=32,mode=mode)
    independent,_=independent_solve(a,.04,eq,geom,target,kind='forced',mode=mode,
                                   reaction_substeps=4,tolerance=1e-12)
    torch.testing.assert_close(actual,forced,atol=2e-11,rtol=2e-7)
    torch.testing.assert_close(actual,independent,atol=2e-11,rtol=2e-7)
    assert actual.dtype==torch.float64


def test_actual_source_quadrature_refines_source_nodes_and_preserves_capped_work():
    a,_,eq,geom=case();path=t.AnchorPath(a,eq,geom,'fd_same_grid')
    actual,meta=t.refined_source_quadrature(path,eq,geom,'fd_same_grid',.04,steps=16,
                                          initial_nodes=2,max_nodes=8,tolerance=1e-10)
    assert meta['accepted'] and meta['fixed_tangent_steps']==16
    assert meta['axis']=='actual_analytic_path_Gauss_Legendre_source_nodes'
    assert meta['unused_path_interpolation'] is False and meta['dense'] is False
    assert meta['source_nodes_evaluated']==sum(row['steps'] for row in meta['refinements'])
    assert meta['homogeneous_rhs_evaluations']==4*16*meta['homogeneous_actions']
    _,capped=t.refined_source_quadrature(path,eq,geom,'fd_same_grid',.04,steps=16,
                                       initial_nodes=2,max_nodes=2,tolerance=1e-10)
    assert not capped['accepted'] and capped['status']=='UNRESOLVED'
    assert capped['uncertainty_rms'] is None and capped['homogeneous_actions']==2
    torch.testing.assert_close(t.source_quadrature(path,eq,geom,'fd_same_grid',0.,nodes=2,steps=2),
                               torch.zeros_like(a),atol=0,rtol=0)


def test_actual_source_quadrature_is_differentiable_in_time_and_physics():
    a,_,_,geom=case()
    k=torch.tensor(.013,dtype=a.dtype,requires_grad=True);r=torch.tensor(2.4,dtype=a.dtype,requires_grad=True)
    h=torch.tensor(.04,dtype=a.dtype,requires_grad=True);eq=n.Physics(k,r)
    path=t.AnchorPath(a,eq,geom,'fd_same_grid',create_graph=True)
    value=t.source_quadrature(path,eq,geom,'fd_same_grid',h,nodes=3,steps=4)
    gradients=torch.autograd.grad(value.square().sum(),(k,r,h))
    assert all(torch.isfinite(g) and g.abs()>0 for g in gradients)


def test_source_quadrature_interrupt_preserves_attempted_action_work():
    a,_,eq,geom=case();path=t.AnchorPath(a,eq,geom,'fd_same_grid')
    class InterruptedBudget:
        calls=0
        def check(self):
            self.calls+=1
            if self.calls>6:raise TimeoutError('Declared test interruption')
    value,meta=t.refined_source_quadrature(path,eq,geom,'fd_same_grid',.04,steps=4,
                                         initial_nodes=2,max_nodes=8,budget=InterruptedBudget())
    assert value is None and meta['status']=='UNRESOLVED'
    assert meta['source_nodes_evaluated']==1 and meta['homogeneous_actions_started']==1
    assert meta['homogeneous_actions']==0 and meta['homogeneous_rhs_evaluations']==12
    assert meta['path_work']['path_queries']==13


@pytest.mark.parametrize('target', TARGETS)
def test_time_and_physics_gradients_and_translation_covariance(target):
    a,z,_,geom=case()
    k=torch.tensor(.013,dtype=a.dtype,requires_grad=True);r=torch.tensor(2.4,dtype=a.dtype,requires_grad=True)
    h=torch.tensor(.04,dtype=a.dtype,requires_grad=True)
    eq=n.Physics(k,r);path=t.AnchorPath(a,eq,geom,target,create_graph=True)
    value=t.response_action(path,eq,geom,target,z,0.,h,steps=4)
    grads=torch.autograd.grad(value.square().sum(),(k,r,h))
    assert all(torch.isfinite(g) and g.abs()>0 for g in grads)
    fixed=n.Physics(.013,2.4);shifted=t.AnchorPath(torch.roll(a,(1,-2),(-2,-1)),fixed,geom,target)
    other=t.response_action(shifted,fixed,geom,target,torch.roll(z,(1,-2),(-2,-1)),0.,.04,steps=4)
    torch.testing.assert_close(other,torch.roll(value.detach(),(1,-2),(-2,-1)),atol=3e-13,rtol=3e-13)


def test_caps_stale_path_unsupported_target_and_dense_production_guard():
    a,z,eq,geom=case();path=t.AnchorPath(a,eq,geom,'fd_same_grid')
    _,meta=t.refined_action(path,eq,geom,'fd_same_grid',z,0.,.08,initial_steps=2,max_steps=2,tolerance=1e-10)
    assert meta['status']=='UNRESOLVED' and not meta['accepted']
    with torch.no_grad():a.add_(.01)
    with pytest.raises(ValueError,match='Stale'):path(.03)
    with pytest.raises(ValueError,match='UNSUPPORTED_TARGET'):t.tangent_action(a,z,eq,geom,'galerkin_projected_fine')
    big=n.Geometry((9,9),(1.,1.));oracle=NumpyEquation(eq,big,'fd_same_grid')
    with pytest.raises(ValueError,match='at most64'):oracle.dense_tangent(np.zeros(big.grid))


def test_smooth_zero_time_path_preserves_right_time_derivative_and_input_jvp():
    a,z,eq,geom=case();a.requires_grad_();h=torch.tensor(0.,dtype=a.dtype,requires_grad=True)
    path=t.AnchorPath(a,eq,geom,'fd_same_grid',create_graph=True)
    value=path(h);torch.testing.assert_close(value,a,atol=0,rtol=0)
    time_slope=torch.autograd.grad((value*z).sum(),h,retain_graph=True)[0]
    torch.testing.assert_close(time_slope,(n.rhs(a,eq,geom,'fd_same_grid')*z).sum(),atol=3e-13,rtol=3e-13)
    input_gradient=torch.autograd.grad(value.sum(),a)[0]
    torch.testing.assert_close(input_gradient,torch.ones_like(a),atol=3e-13,rtol=3e-13)
    h2=torch.tensor(0.,dtype=a.dtype,requires_grad=True)
    action=t.response_action(path,eq,geom,'fd_same_grid',z,0.,h2,steps=2)
    slope=torch.autograd.grad(action.square().sum(),h2)[0]
    expected=2*(z*t.tangent_action(a,z,eq,geom,'fd_same_grid')).sum()
    torch.testing.assert_close(slope,expected,atol=3e-12,rtol=3e-12)


def test_teacher_combined_budget_actual_input_and_capped_unresolved():
    a,_,eq,geom=case()
    actual=a.float();teacher,meta=refined_teacher(actual,.04,eq,geom,'fd_same_grid',config=dict(initial_steps=8,max_steps=64,tolerance=1e-10))
    assert teacher.dtype==torch.float64 and meta['accepted']
    assert meta['uncertainty_rms']<=1e-10 and meta['uncertainty_maximum']<=1e-10
    _,capped=refined_teacher(actual,.04,eq,geom,'fd_same_grid',config=dict(initial_steps=2,max_steps=2,tolerance=1e-10))
    assert capped['status']=='UNRESOLVED' and not capped['accepted']


@pytest.mark.parametrize('target', TARGETS)
def test_finite_product_constant_and_signed_bound_terms_keep_scope(target):
    a,z,eq,geom=case(6)
    constant=t.finite_product_constant(geom,target)
    product=n.product(a,z,geom,target)
    lhs=n.inner(product,product,geom,target).sqrt()
    rhs=constant*n.inner(a,a,geom,target).sqrt()*n.inner(z,z,geom,target).sqrt()
    assert (lhs<=rhs).all()
    decomposition=t.signed_error_decomposition(z,-.8*z,geom,target)
    assert decomposition['signed_cross_term']<0 and abs(decomposition['algebraic_gap'])<1e-14
    bound=t.conditional_first_correction_bound(.1,.2,eq,geom,target,quadrature_error=.01)
    assert bound['bound']>.11 and not bound['certified'] and not bound['grid_uniform']
    growth=t.conditional_growth_bound(eq,.1,anchor_lower_bound=.2)
    assert growth['growth']>1 and not growth['certified']
