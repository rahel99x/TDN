"""Independent tiny FD controls; no native or scientific efficiency claim."""
import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from tdn.analysis.response import numerics as n
from tdn.analysis.tangent_response.tangent import AnchorPath
from tdn.analysis.response_preconditioner import numerics as p


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def case(grid=(5,)):
    geom = n.Geometry(grid, tuple(1.+.3*j for j in range(len(grid))))
    x = torch.arange(np.prod(grid), dtype=torch.float64).reshape(1, 1, *grid)
    state = .3+.07*torch.cos(1.3*x)
    source = torch.sin(1.7*x)+.2
    return state, source, n.Physics(.023, 1.4), geom


class PolynomialPath:
    def __init__(self, state, slope=None, curvature=None):
        self.state = state
        self.slope = torch.zeros_like(state) if slope is None else slope
        self.curvature = torch.zeros_like(state) if curvature is None else curvature

    def __call__(self, time):
        time = n.scalar(time, self.state)
        return self.state+time*self.slope+time.square()*self.curvature

    def evaluate(self, time):
        return {'state': self(time), 'derivative': self.slope+2*n.scalar(time, self.state)*self.curvature}


def dense_diffusion(eq, geom):
    """Independent NumPy stencil assembly, without production generator/FFT."""
    count = int(np.prod(geom.grid))
    columns = []
    for index in range(count):
        basis = np.zeros(geom.grid)
        basis.flat[index] = 1.
        action = np.zeros(geom.grid)
        for axis, dx in enumerate(geom.dx):
            action += (np.roll(basis, -1, axis)-2*basis+np.roll(basis, 1, axis))/dx**2
        columns.append(float(eq.kappa)*action.ravel())
    return np.stack(columns, axis=1)


@pytest.mark.parametrize('grid', ((5,), (3, 4)))
def test_static_response_refines_to_independent_dense_exponential(grid):
    state, source, eq, geom = case(grid)
    matrix = dense_diffusion(eq, geom)+np.diag(eq.reaction_rate*(1-2*state.numpy().ravel()))
    truth = expm(.23*matrix)@source.numpy().ravel()
    errors = [np.linalg.norm(p.ordered_response(PolynomialPath(state), source, .03, .26,
                                               eq, geom, panels=panels).numpy().ravel()-truth)
              for panels in (1, 2, 4, 8)]
    assert errors[-1] < 1e-4
    assert all(3.7 < errors[i]/errors[i+1] < 4.3 for i in range(3))


def test_nonautonomous_order_matches_dense_panels_and_refined_ivp():
    state, source, eq, geom = case()
    slope = .8*torch.sin(torch.arange(5, dtype=state.dtype)).reshape_as(state)
    path = PolynomialPath(state, slope)
    start, stop = .03, .41
    laplace = dense_diffusion(eq, geom)
    dt = (stop-start)/2
    factors = []
    for panel in range(2):
        left, right = start+panel*dt, start+(panel+1)*dt
        integrated_state = dt*state.numpy().ravel()+.5*(right**2-left**2)*slope.numpy().ravel()
        half = np.diag(np.exp(.5*eq.reaction_rate*(dt-2*integrated_state)))
        factors.append(half@expm(dt*laplace)@half)
    expected = factors[1]@factors[0]@source.numpy().ravel()
    actual = p.ordered_response(path, source, start, stop, eq, geom)
    np.testing.assert_allclose(actual.numpy().ravel(), expected, atol=8e-15, rtol=8e-15)
    reverse = factors[0]@factors[1]@source.numpy().ravel()
    assert np.linalg.norm(reverse-expected) > 1e-3
    reference = solve_ivp(lambda time, z: laplace@z+eq.reaction_rate*(
        1-2*(state.numpy().ravel()+time*slope.numpy().ravel()))*z,
        (start, stop), source.numpy().ravel(), method='DOP853', rtol=2e-13, atol=2e-14)
    assert reference.success
    truth = reference.y[:, -1]
    errors = [np.linalg.norm(p.ordered_response(path, source, start, stop, eq, geom,
                panels=panels).numpy().ravel()-truth) for panels in (1, 2, 4, 8)]
    assert all(3.5 < errors[i]/errors[i+1] < 4.5 for i in range(3))


@pytest.mark.parametrize('limit', ('heat', 'static_uniform', 'local_reaction', 'moving_uniform'))
def test_analytic_response_limits(limit):
    state, source, eq, geom = case()
    if limit == 'heat':
        eq = n.Physics(eq.kappa, 0.)
    if limit.endswith('uniform'):
        state = torch.full_like(state, .29)
    slope = torch.full_like(state, .07) if limit == 'moving_uniform' else .1*state
    curvature = .08*state
    if limit == 'static_uniform':
        slope, curvature = torch.zeros_like(state), torch.zeros_like(state)
    if limit == 'local_reaction':
        eq = n.Physics(0., eq.reaction_rate)
    start, stop = .04, .27
    path = PolynomialPath(state, slope, curvature)
    integral = ((stop-start)*state+.5*(stop**2-start**2)*slope+
                (stop**3-start**3)*curvature/3)
    multiplier = torch.exp(eq.reaction_rate*((stop-start)-2*integral)).numpy().ravel()
    expected = multiplier*(expm((stop-start)*dense_diffusion(eq, geom))@source.numpy().ravel())
    actual = p.ordered_response(path, source, start, stop, eq, geom)
    np.testing.assert_allclose(actual.numpy().ravel(), expected, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize('grid', ((5,), (3, 4)))
def test_seven_actual_stencils_equal_independent_dense_commutators(grid, monkeypatch):
    state, source, eq, geom = case(grid)
    slope, curvature = .4*state, .2*state.square()
    path = PolynomialPath(state, slope, curvature)
    calls = []
    original = n.generator
    monkeypatch.setattr(n, 'generator', lambda *args: (calls.append(1), original(*args))[1])
    work = {'laplacian_actions': 9, 'unrelated': 2}
    actual = p.commutator_actions(path, source, .08, eq, geom, work=work)
    laplace = dense_diffusion(eq, geom)
    potential = np.diag(eq.reaction_rate*(1-2*path(.08).numpy().ravel()))
    derivative = np.diag(-2*eq.reaction_rate*path.evaluate(.08)['derivative'].numpy().ravel())
    comm = lambda a, b: a@b-b@a
    matrices = (comm(laplace, comm(laplace, potential)),
                comm(potential, comm(laplace, potential)), comm(laplace, derivative))
    for action, matrix in zip(actual, matrices):
        np.testing.assert_allclose(action.numpy().ravel(), matrix@source.numpy().ravel(), atol=7e-15, rtol=5e-13)
    assert len(calls) == 7
    assert work == {'laplacian_actions': 16, 'unrelated': 2,
                    'potential_path_queries': 1, 'derivative_path_queries': 1}


@pytest.mark.parametrize('limit', ('zero_diffusion', 'zero_reaction', 'homogeneous'))
def test_commutator_nulls(limit):
    state, source, eq, geom = case()
    if limit == 'zero_diffusion':
        eq = n.Physics(0., eq.reaction_rate)
    if limit == 'zero_reaction':
        eq = n.Physics(eq.kappa, 0.)
    if limit == 'homogeneous':
        state = torch.full_like(state, .3)
    values = p.commutator_actions(PolynomialPath(state, .1*state), source, .02, eq, geom)
    for value in values:
        torch.testing.assert_close(value, torch.zeros_like(value), atol=2e-15, rtol=0)


def test_actual_path_derivative_used_and_complete_gradients_preserved():
    state, source, _, geom = case((3,))
    state.requires_grad_()
    source.requires_grad_()
    kappa = torch.tensor(.023, dtype=state.dtype, requires_grad=True)
    rate = torch.tensor(1.4, dtype=state.dtype, requires_grad=True)
    start = torch.tensor(.03, dtype=state.dtype, requires_grad=True)
    stop = torch.tensor(.11, dtype=state.dtype, requires_grad=True)

    def action(u, z, k, r, s, t):
        eq = n.Physics(k, r)
        path = AnchorPath(u, eq, geom, 'fd_same_grid', create_graph=True)
        ordered = p.ordered_response(path, z, s, t, eq, geom)
        features = p.commutator_actions(path, z, s, eq, geom)
        return torch.cat((ordered.flatten(), *(feature.flatten() for feature in features)))

    assert torch.autograd.gradcheck(action, (state, source, kappa, rate, start, stop),
                                    eps=1e-6, atol=2e-7, rtol=2e-5, fast_mode=True)


def test_two_sided_heat_matches_dense_placement_and_all_gradients():
    state, source, eq, geom = case()
    b, c = 1+.2*state, .7+state.square()
    duration = .14
    actual = p.two_sided_heat(source, b, c, duration, eq, geom)
    expected = b.numpy().ravel()*(expm(duration*dense_diffusion(eq, geom))@(c*source).numpy().ravel())
    np.testing.assert_allclose(actual.numpy().ravel(), expected, atol=2e-15, rtol=5e-15)
    incorrect = b*c*n.heat(source, duration, eq, geom, 'fd_same_grid')
    assert float((actual-incorrect).abs().max()) > 1e-4
    kappa = torch.tensor(.023, dtype=state.dtype, requires_grad=True)
    time = torch.tensor(duration, dtype=state.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z, output, incoming, k, h:
        p.two_sided_heat(z, output, incoming, h, n.Physics(k, 0.), geom),
        (source.requires_grad_(), b.requires_grad_(), c.requires_grad_(), kappa, time),
        eps=1e-6, atol=2e-8, rtol=2e-6, fast_mode=True)


def test_zero_duration_is_literal_identity_with_right_time_and_input_derivative():
    state, source, eq, geom = case()
    path = PolynomialPath(state, .1*state)
    delta = torch.tensor(0., dtype=source.dtype, requires_grad=True)
    output, derivative = torch.autograd.functional.jvp(
        lambda h: p.ordered_response(path, source, .04, .04+h, eq, geom),
        delta, torch.ones_like(delta))
    torch.testing.assert_close(output, source, atol=0, rtol=0)
    expected = n.generator(source, eq, geom, 'fd_same_grid')+eq.reaction_rate*(1-2*path(.04))*source
    torch.testing.assert_close(derivative, expected, atol=3e-15, rtol=3e-15)
    _, source_derivative = torch.autograd.functional.jvp(
        lambda z: p.ordered_response(path, z, .04, .04, eq, geom), source, state)
    torch.testing.assert_close(source_derivative, state, atol=3e-15, rtol=3e-15)
    b, c = 1+state, 1-state
    value, heat_derivative = torch.autograd.functional.jvp(
        lambda h: p.two_sided_heat(source, b, c, h, eq, geom), delta, torch.ones_like(delta))
    torch.testing.assert_close(value, b*c*source, atol=0, rtol=0)
    torch.testing.assert_close(heat_derivative, b*n.generator(c*source, eq, geom, 'fd_same_grid'), atol=3e-15, rtol=3e-15)


def test_source_linearity_complete_work_and_parent_batching():
    state, source, eq, geom = case()
    path = PolynomialPath(state, .1*state)
    actions = (lambda z: p.ordered_response(path, z, .02, .13, eq, geom),
               lambda z: torch.stack(p.commutator_actions(path, z, .02, eq, geom)),
               lambda z: p.two_sided_heat(z, 1+state, 1-state, .11, eq, geom))
    for action in actions:
        torch.testing.assert_close(action(.3*source-.8*state), .3*action(source)-.8*action(state), atol=4e-15, rtol=2e-13)
    work = {'heat_actions': 4}
    p.ordered_response(path, source, .02, .13, eq, geom, work=work)
    p.two_sided_heat(source, 1+state, 1-state, .11, eq, geom, work=work)
    assert work == {'heat_actions': 7, 'fft_transforms': 6, 'potential_path_queries': 4}
    both_state = torch.cat((state, .8*state))
    both_source = torch.cat((source, 2*source))
    batch_eq = n.Physics(torch.tensor([.023, .041], dtype=state.dtype),
                         torch.tensor([1.4, .7], dtype=state.dtype))
    times = torch.tensor([0., .13], dtype=state.dtype)
    actual = p.ordered_response(PolynomialPath(both_state), both_source, 0., times, batch_eq, geom)
    torch.testing.assert_close(actual[:1], source, atol=0, rtol=0)
    expected = p.ordered_response(PolynomialPath(.8*state), 2*source, 0., .13, n.Physics(.041, .7), geom)
    torch.testing.assert_close(actual[1:], expected, atol=3e-15, rtol=3e-15)


@pytest.mark.parametrize('action', ('ordered', 'commutator', 'heat'))
def test_unsupported_targets_and_nonfinite_fields_rejected(action):
    state, source, eq, geom = case()
    path = PolynomialPath(state)
    def call(z, target='fd_same_grid'):
        if action == 'ordered':
            return p.ordered_response(path, z, 0., .1, eq, geom, target=target)
        if action == 'commutator':
            return p.commutator_actions(path, z, 0., eq, geom, target=target)
        return p.two_sided_heat(z, 1+state, 1-state, .1, eq, geom, target=target)
    with pytest.raises(ValueError, match='UNSUPPORTED_TARGET'):
        call(source, 'galerkin_same_grid')
    with pytest.raises(ValueError, match='finite'):
        call(source*float('nan'))
    with pytest.raises(ValueError, match='geometry'):
        call(source[..., :-1])


def test_invalid_intervals_budgets_factors_paths_and_detached_graph_rejected():
    state, source, eq, geom = case()
    path = PolynomialPath(state)
    for start, stop in ((.2, .1), (0., float('inf')), (float('nan'), .1), (-.1, .1)):
        with pytest.raises(ValueError):
            p.ordered_response(path, source, start, stop, eq, geom)
    for kwargs in ({'panels': 3}, {'panels': True}, {'potential_nodes': 5}, {'potential_nodes': False}):
        with pytest.raises(ValueError):
            p.ordered_response(path, source, 0., .1, eq, geom, **kwargs)
    with pytest.raises(ValueError, match='match'):
        p.two_sided_heat(source, state.float(), state, .1, eq, geom)
    with pytest.raises(ValueError, match='finite'):
        p.two_sided_heat(source, state*float('inf'), state, .1, eq, geom)
    with pytest.raises(ValueError, match='finite'):
        p.ordered_response(lambda time: state*float('nan'), source, 0., .1, eq, geom)
    with pytest.raises(ValueError, match='evaluate'):
        p.commutator_actions(lambda time: state, source, 0., eq, geom)
    detached_path = AnchorPath(state.requires_grad_(), eq, geom, 'fd_same_grid')
    with pytest.raises(ValueError, match='create_graph=True'):
        p.ordered_response(detached_path, source, 0., .1, eq, geom)
    with pytest.raises(ValueError, match='create_graph=True'):
        p.commutator_actions(detached_path, source, 0., eq, geom)


def test_anchor_physics_must_match_values_and_live_derivative_identity():
    state, source, eq, geom = case()
    path = AnchorPath(state, eq, geom, 'fd_same_grid')
    for bad_eq in (n.Physics(.031, eq.reaction_rate), n.Physics(eq.kappa, 2.)):
        with pytest.raises(ValueError, match='match response physics'):
            p.ordered_response(path, source, 0., .1, bad_eq, geom)
        with pytest.raises(ValueError, match='match response physics'):
            p.commutator_actions(path, source, 0., bad_eq, geom)
    kappa = torch.tensor(.023, dtype=state.dtype, requires_grad=True)
    independent_kappa = kappa.detach().clone().requires_grad_()
    differentiable_eq = n.Physics(kappa, eq.reaction_rate)
    path = AnchorPath(state, differentiable_eq, geom, 'fd_same_grid', create_graph=True)
    with pytest.raises(ValueError, match='same tensors'):
        p.ordered_response(path, source, 0., .1, n.Physics(independent_kappa, eq.reaction_rate), geom)
    # Equal immutable scalar copies do not introduce an ambiguous gradient.
    scalar_path = AnchorPath(state, eq, geom, 'fd_same_grid')
    actual = p.ordered_response(scalar_path, source, 0., .1, n.Physics(.023, 1.4), geom)
    assert torch.isfinite(actual).all()
