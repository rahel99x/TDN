"""Independent identities and gradient regressions for the Response equation."""
import numpy as np
import pytest
import torch
from scipy.signal import resample as scipy_resample
from tdn.numerics.types import Geometry, Equation
from tdn.analysis.response.contracts import Physics, TargetContract, PrecisionContract
from tdn.analysis.response import numerics as n
from tdn.analysis.response import kernels as k
from tdn.analysis.response import trajectory as tr

DT = torch.float64


def field(grid=(9,), batch=1):
    gen = torch.Generator().manual_seed(83041)
    return .37+.025*torch.randn(batch, 1, *grid, dtype=DT, generator=gen)


@pytest.mark.parametrize('grid', [(15,), (16,), (9, 9), (10, 10)])
@pytest.mark.parametrize('target', ['fd_same_grid', 'galerkin_same_grid'])
def test_response_logistic_transform_mean_variance_and_error_identity(grid, target):
    geom = Geometry(grid, (1.,)*len(grid)); eq = Physics(.006, 3.)
    w = field(grid)-.37
    bg = k.background(torch.full_like(n.mean(w), .37), .12, 3.)
    u = bg['c']+bg['q']*w
    transformed = bg['c_derivative']+bg['q_derivative']*w+bg['q']*(n.generator(w, eq, geom, target)-eq.reaction_rate*bg['q']*n.product(w,w,geom,target))
    torch.testing.assert_close(n.rhs(u,eq,geom,target), transformed, rtol=2e-12,atol=3e-13)
    moments = tr.moment_rhs(u,eq,geom,target)
    torch.testing.assert_close(n.mean(n.rhs(u,eq,geom,target)),moments['mean_derivative'],rtol=2e-12,atol=2e-13)
    direct_variance = 2*n.inner(n.center(u),n.rhs(u,eq,geom,target),geom,target)
    torch.testing.assert_close(direct_variance,moments['variance_derivative'],rtol=2e-12,atol=2e-13)
    a = field(grid); da = torch.sin(a)
    identity = tr.full_error_identity(u,a,da,eq,geom,target)
    assert identity['difference'].abs().max() < 2e-12


@pytest.mark.parametrize('grid', [(8,), (8,10)])
def test_response_nyquist_physical_mass_and_independent_projection(grid):
    geom=Geometry(grid,(1.,)*len(grid)); u=field(grid)
    for axis in range(2,u.ndim):
        shape=[1]*u.ndim;shape[axis]=grid[axis-2]
        u=u+.2*(-1.)**torch.arange(grid[axis-2],dtype=DT).reshape(shape)
    fine=tuple(3*x+1 for x in grid)
    lifted=n.resample(u,fine)
    expected=u.numpy()
    for axis,size in enumerate(fine,2):expected=scipy_resample(expected,size,axis=axis)
    np.testing.assert_allclose(lifted.numpy(),expected,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(n.inner(u,u,geom,'galerkin_same_grid'),n.mean(lifted.square()),rtol=1e-13,atol=1e-13)
    assert abs(float(n.inner(u,u,geom,'galerkin_same_grid')-n.mean(u.square())))>.01
    torch.testing.assert_close(n.mean(n.product(u,u,geom,'galerkin_same_grid')),n.mean(lifted.square()),rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('target', ['fd_same_grid','galerkin_same_grid'])
def test_response_shared_quadrature_parity_and_mean_spectral_sum(target):
    from tdn.analysis.resolve.kernels import batched_quadratic_defect
    u=field((8,8),batch=2);geom=Geometry((8,8),(1.,1.));eq=Physics(.006,3.)
    q=n.quadratic_defect(u,.12,eq,geom,target,nodes=4)
    old=batched_quadratic_defect(u,.12,Equation(.006,3.),geom,track='discrete' if target.startswith('fd') else 'continuum',nodes=4)
    torch.testing.assert_close(q,old,rtol=2e-10,atol=2e-16)
    torch.testing.assert_close(q,n.quadratic_defect(u,.12,eq,geom,target,nodes=4,shared=False),rtol=2e-10,atol=2e-16)
    torch.testing.assert_close(n.mean(q),k.mean_quadratic_spectral(u,.12,eq,geom,target,nodes=4),rtol=1e-10,atol=1e-16)


@pytest.mark.parametrize('target', ['fd_same_grid','galerkin_same_grid'])
@pytest.mark.parametrize('which', ['df','shared_gl4'])
def test_response_anchor_physics_time_and_state_gradients(target,which):
    u=field((5,)).requires_grad_();geom=Geometry((5,),(1.,))
    kap=torch.tensor(.006,dtype=DT,requires_grad=True);rate=torch.tensor(3.,dtype=DT,requires_grad=True);h=torch.tensor(.07,dtype=DT,requires_grad=True)
    def call(u,kap,rate,h):return n.anchor(u,h,Physics(kap,rate),geom,target,which,reaction_substeps=4)
    assert torch.autograd.gradcheck(call,(u,kap,rate,h),eps=1e-6,atol=2e-7,rtol=2e-5)
    zero=torch.tensor(0.,dtype=DT,requires_grad=True)
    grad=torch.autograd.grad(n.heat(u,.02,Physics(zero,rate),geom,target).square().sum(),zero)[0]
    assert grad.abs()>1e-7


@pytest.mark.parametrize('target', ['fd_same_grid','galerkin_same_grid'])
def test_response_anchor_path_derivative_and_full_coupled_source(target):
    u=field((7,));geom=Geometry((7,),(1.,));eq=Physics(.006,3.)
    t=.12;delta=1e-6
    p=tr.anchor_path(u,t,eq,geom,target,reaction_substeps=8)
    diff=(n.anchor(u,t+delta,eq,geom,target)-n.anchor(u,t-delta,eq,geom,target))/(2*delta)
    torch.testing.assert_close(p['derivative'],diff,rtol=2e-8,atol=3e-10)
    truth=n.integrate(u,t,eq,geom,target,substeps=32)
    source=tr.actual_source(truth,p['state'],p['derivative'],eq,geom,target)
    d=source['spatial_error']; m=source['mean']
    direct=n.center(n.rhs(truth,eq,geom,target)-p['derivative'])
    expanded=n.generator(d,eq,geom,target)+eq.reaction_rate*(1-2*m)*d+source['source']
    torch.testing.assert_close(direct,expanded,rtol=1e-9,atol=1e-13)
    assert n.mean(source['source']).abs().max()<1e-17


def test_response_source_response_matches_independent_single_mode_integral():
    from scipy.integrate import quad
    geom=Geometry((15,),(2.,));eq=Physics(.03,2.)
    x=torch.arange(15,dtype=DT)*2/15;v=torch.cos(torch.pi*x)[None,None,:];h=.2
    source=lambda t:(1+t)*v
    meanpath=lambda t:.3+.1*t+torch.zeros_like(n.mean(v))
    result=tr.source_response(source,h,eq,geom,'galerkin_same_grid',meanpath,nodes=16)
    expected=quad(lambda s:(1+s)*np.exp(-.03*np.pi**2*(h-s)+2*(.4*(h-s)-.1*(h*h-s*s))),0,h,epsabs=1e-14)[0]*v
    torch.testing.assert_close(result,expected,rtol=2e-13,atol=2e-14)


def test_response_stable_kernels_limits_and_representable_gradients():
    for order in (1,2,3):
        z=torch.tensor([0.,-1e-8,-.1,-1000.],dtype=DT,requires_grad=True)
        value=k.phi(z,order)
        assert torch.isfinite(value).all()
        assert torch.isfinite(torch.autograd.grad(value.sum(),z)[0]).all()
    x=torch.tensor([0.,1e-10,1e-6,.01],dtype=DT,requires_grad=True)
    value=k.quadratic_mean_kernel(x)
    torch.testing.assert_close(value[1:3]/x[1:3]**2,torch.full_like(x[1:3],1/6),rtol=2e-6,atol=0)
    assert torch.isfinite(torch.autograd.grad(value.sum(),x)[0]).all()
    c=torch.tensor([0.,1.,.37],dtype=DT).reshape(3,1,1)
    b=k.background(c,1000.,1.)
    assert b['c'][0]==0 and b['c'][1]==1 and torch.isinf(b['q'][0])
    ratio=k.q_ratio(c,1000.,999.,1.)
    torch.testing.assert_close(ratio[:2].flatten(),torch.tensor([np.e,1/np.e],dtype=DT))
    with pytest.raises(ValueError):k.background(torch.tensor(-.1),.1,3.)


def test_response_batch_physics_is_not_spatial_broadcast_or_detached():
    u=field((7,),batch=2);geom=Geometry((7,),(1.,))
    kap=torch.tensor([.01,.02],dtype=DT,requires_grad=True);rate=torch.tensor([2.,3.],dtype=DT,requires_grad=True)
    together=n.anchor(u,torch.tensor([.03,.04],dtype=DT),Physics(kap,rate),geom,'fd_same_grid','shared_gl4')
    separate=torch.cat([n.anchor(u[i:i+1],(.03,.04)[i],Physics(kap[i],rate[i]),geom,'fd_same_grid','shared_gl4') for i in range(2)])
    torch.testing.assert_close(together,separate,rtol=2e-13,atol=2e-14)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(together.square().sum(),(kap,rate)))


def test_response_precision_and_target_contracts_do_not_conflate_rounding_or_closure():
    with pytest.raises(ValueError):TargetContract('galerkin_projected_fine',(8,),(1.,))
    with pytest.raises(ValueError):PrecisionContract(kind='deployment_defect')
    teacher=field((7,));physical=teacher+.000001;deployed=physical.float()
    labels=PrecisionContract().labels(teacher,physical,deployed)
    torch.testing.assert_close(labels['deployment_defect'],labels['physical_defect']+labels['anchor_rounding'])
    with pytest.raises(ValueError):n.rhs(teacher,Physics(.01,3.),Geometry((7,),(1.,)),'continuum')


@pytest.mark.parametrize('target', ['fd_same_grid','galerkin_same_grid'])
def test_response_eigenmodes_temporal_order_and_etdrk4_against_independent_ode(target):
    from scipy.integrate import solve_ivp
    geom=Geometry((9,),(2.,));eq=Physics(.012,3.)
    x=torch.arange(9,dtype=DT)*2/9
    wave=torch.cos(2*torch.pi*x)[None,None,:]
    eigenvalue=-.012*(4*9**2/4*np.sin(2*np.pi/9)**2 if target=='fd_same_grid' else (2*np.pi)**2)
    torch.testing.assert_close(n.generator(wave,eq,geom,target),eigenvalue*wave,rtol=1e-13,atol=2e-14)
    torch.testing.assert_close(n.heat(wave,.17,eq,geom,target),np.exp(eigenvalue*.17)*wave,rtol=1e-13,atol=2e-14)
    initial=(.4+.1*wave).numpy().ravel()
    def rhs_np(t,u):
        if target=='fd_same_grid':return .012*(np.roll(u,1)-2*u+np.roll(u,-1))/(2/9)**2+3*u*(1-u)
        symbol=-.012*(2*np.pi*np.fft.fftfreq(9,d=2/9))**2
        square=scipy_resample(scipy_resample(u,27)**2,9)
        return np.fft.ifft(np.fft.fft(u)*symbol).real+3*(u-square)
    truth=solve_ivp(rhs_np,(0,.2),initial,method='DOP853',rtol=3e-14,atol=1e-15).y[:,-1]
    errors=[]
    for count in (2,4,8):
        u=torch.tensor(initial,dtype=DT)[None,None,:]
        for _ in range(count):u=n.etdrk4_step(u,.2/count,eq,geom,target)
        errors.append(np.sqrt(np.mean((u.numpy().ravel()-truth)**2)))
    assert min(np.log2(np.array(errors[:-1])/np.array(errors[1:])))>3.8


def test_response_galerkin_nesting_and_high_to_low_interactions_cannot_be_compressed_away():
    N=17;g=Geometry((N,),(1.,));x=torch.arange(N,dtype=DT)/N
    u=torch.cos(2*torch.pi*6*x)[None,None,:]
    v=torch.cos(2*torch.pi*7*x)[None,None,:]
    w=torch.cos(2*torch.pi*2*x)[None,None,:]
    B=lambda a,b:n.product(a,b,g,'galerkin_same_grid')
    # Truncated Fourier product is not associative; preserve nested causal algebra.
    assert (B(B(u,v),w)-B(u,B(v,w))).abs().max()>.1
    projected=n.resample(u,(9,));square_then_project=n.resample(B(u,u),(9,))
    assert projected.abs().max()<1e-14
    torch.testing.assert_close(square_then_project,torch.full_like(square_then_project,.5),rtol=1e-13,atol=1e-13)


def test_response_quadratic_spectral_resonance_and_small_amplitude_coefficients():
    from scipy.integrate import solve_ivp
    N=9;geom=Geometry((N,N),(1.,1.));x=torch.arange(N,dtype=DT)/N
    a=torch.cos(2*torch.pi*x)[:,None].expand(N,N)[None,None]
    b=torch.cos(2*torch.pi*x)[None,:].expand(N,N)[None,None]
    eq=Physics(.01,3.)
    qboth=n.quadratic_defect(.4+a+b,.12,eq,geom,'galerkin_same_grid')
    qsingle=n.quadratic_defect(.4+a,.12,eq,geom,'galerkin_same_grid')+n.quadratic_defect(.4+b,.12,eq,geom,'galerkin_same_grid')
    # Orthogonal wavevectors satisfy lambda_(p+q)=lambda_p+lambda_q.
    torch.testing.assert_close(qboth,qsingle,rtol=1e-10,atol=1e-14)
    # Fixed finite h, separate amplitude expansion: Q removes the epsilon^2 term.
    N=9;geom=Geometry((N,),(1.,));x=np.arange(N)/N;shape=np.cos(2*np.pi*x)+.3*np.cos(4*np.pi*x+.4)
    errors=[]
    for amplitude in (.02,.01,.005):
        init=.4+amplitude*shape
        def fun(t,u):return .01*N*N*(np.roll(u,1)-2*u+np.roll(u,-1))+3*u*(1-u)
        truth=solve_ivp(fun,(0,.12),init,method='DOP853',rtol=3e-14,atol=1e-15).y[:,-1]
        tensor=torch.tensor(init,dtype=DT)[None,None,:]
        out=n.anchor(tensor,.12,eq,geom,'fd_same_grid','gl16').numpy().ravel()
        errors.append(np.sqrt(np.mean((truth-out)**2)))
    assert min(np.log2(np.array(errors[:-1])/np.array(errors[1:])))>2.8


def test_response_physical_label_can_disappear_in_fp32_deployed_addition():
    anchor=torch.ones(1,1,5,dtype=torch.float32)
    correction=torch.full_like(anchor,1e-9)
    assert torch.equal(anchor+correction,anchor)
    high=anchor.double()+correction.double()
    assert (high-anchor.double()).abs().min()>0


def test_response_independent_appendix_audit_records_all_actual_probes(tmp_path):
    from types import SimpleNamespace
    from tdn.analysis.response.audit import run
    from tdn.analysis.premix.neural import _RunBudget
    rows=[]
    ctx=SimpleNamespace(path=tmp_path,budget=_RunBudget(120,None,'cpu'),
        record=lambda name,mechanisms,**kwargs:rows.append((name,kwargs)))
    summary=run(ctx)
    assert summary['appendix_sections']==['A'+str(i) for i in range(1,10)]
    assert summary['designed_A7_cases']==54
    assert not [(name,c) for name,row in rows for c in row['checks'] if c['verdict']=='BAD']
    # The conclusion is produced by independently simulated phase twins, not a constant assertion.
    phase=next(row for name,row in rows if name=='A5/phase_twins')['metrics']['fields']
    assert phase[0]['remaining_mean_defect']>1e-4 and phase[1]['remaining_mean_defect'] < -3e-4


def test_response_projected_fine_forcing_cannot_be_silently_called_coarse_dynamics():
    u=field((7,));geom=Geometry((7,),(1.,));eq=Physics(.003,3.)
    a=u+.01*torch.cos(torch.arange(7,dtype=DT))[None,None,:]
    da=n.rhs(a,eq,geom,'galerkin_same_grid')
    with pytest.raises(ValueError,match='closure'):
        tr.actual_source(u,a,da,eq,geom,'galerkin_projected_fine')
    with pytest.raises(ValueError,match='closure'):
        tr.full_error_identity(u,a,da,eq,geom,'galerkin_projected_fine')
    forcing=torch.sin(u)
    identity=tr.full_error_identity(u,a,da,eq,geom,'galerkin_projected_fine',closure_forcing=forcing)
    assert identity['difference'].abs().max()<1e-13
    source=tr.actual_source(u,a,da,eq,geom,'galerkin_projected_fine',closure_forcing=forcing)
    unforced=tr.actual_source(u,a,da,eq,geom,'galerkin_same_grid')
    torch.testing.assert_close(source['source']-unforced['source'],n.center(forcing),atol=1e-13,rtol=1e-13)


def test_response_clock_validates_information_and_time_weights():
    m=torch.tensor([[[.3]]],dtype=DT)
    out=tr.conditional_clock(m,[m,m],[m*0,m*0],3.,[.03,.07])
    torch.testing.assert_close(out,n.logistic(m,.1,3.),rtol=1e-14,atol=1e-14)
    with pytest.raises(ValueError,match='matching'):
        tr.conditional_clock(m,[m],[m*0,m*0],3.,[.1])
    with pytest.raises(ValueError,match='Variance'):
        tr.conditional_clock(m,[m],[torch.ones_like(m)],3.,[.1])


@pytest.mark.parametrize('target',['fd_same_grid','galerkin_same_grid'])
def test_response_path_jvp_is_correct_inside_inference_mode(target):
    u=field((7,));geom=Geometry((7,),(1.,));eq=Physics(.01,3.)
    normal=tr.anchor_path(u,.12,eq,geom,target,create_graph=False)
    with torch.inference_mode():
        inference_u=u.clone()
        physics=Physics(torch.tensor(.01,dtype=DT),torch.tensor(3.,dtype=DT))
        inside=tr.anchor_path(inference_u,torch.tensor(.12,dtype=DT),physics,geom,target,create_graph=False)
        teacher=tr.reference_path(inference_u,.12,physics,geom,target,substeps=8,create_graph=False)
    torch.testing.assert_close(inside['derivative'],normal['derivative'],rtol=1e-13,atol=1e-14)
    assert inside['derivative'].abs().max()>.1
    assert teacher['derivative'].abs().max()>.1
