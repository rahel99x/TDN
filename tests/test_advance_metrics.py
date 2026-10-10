import json
import math

import pytest
import torch

from tdn.analysis.advance.metrics import endpoint_metrics, measure_paired_metrics, energy_value, physical_inner_product
from tdn.numerics import Equation, Geometry


def field(n=16, length=1., mode=3):
    x=torch.arange(n,dtype=torch.float64)/n
    return (.4+.05*torch.cos(2*math.pi*mode*x)[:,None].expand(n,n))[None,None],Geometry((n,n),(length,length))


@pytest.mark.parametrize('track',['discrete','continuum'])
def test_error_mean_centered_and_spectral_partitions(track):
    u,g=field();prediction=u+.02+.01*torch.sin(2*math.pi*torch.arange(16,dtype=u.dtype)/16)[None,None,:,None]
    out=endpoint_metrics(prediction,u,u,Equation(.01,2),g,track,dict(accepted=True,uncertainty_rms=1e-8,uncertainty_max_bound=2e-8))
    assert out['mean_error_rms']==pytest.approx(.02)
    assert out['centered_rms']==pytest.approx(.01/math.sqrt(2))
    assert abs(out['error_decomposition_residual'])<1e-18
    assert sum(b['error_rms']**2 for b in out['spectral_bands']if b['error_rms']is not None)==pytest.approx(out['error_rms']**2)
    assert out['upper_rms']==pytest.approx(out['error_rms']+1e-8)
    assert out['upper_max']==pytest.approx(out['error_max']+2e-8)
    assert out['mean_balance_residual'] is None
    json.dumps(out,allow_nan=False)


def test_fixed_physical_bands_are_grid_and_domain_aware():
    errors=[]
    for n in (16,32,64):
        u,g=field(n,length=2.,mode=3)
        target=torch.full_like(u,.4)
        out=endpoint_metrics(u,target,target,Equation(.01,1),g,'continuum')
        bands={b['label']:b for b in out['spectral_bands']}
        assert bands['low']['error_rms']==pytest.approx(.05/math.sqrt(2)) # 3 / 2 = 1.5 cycles / length
        assert bands['middle']['error_rms']<1e-15
        errors.append(out['prediction']['gradient_squared_mean'])
    assert errors==pytest.approx([(.05**2)/2*(3*math.pi)**2]*3)


def test_discrete_gradient_has_exact_fd_energy_and_nyquist_content():
    n=16;u,g=field(n,mode=n//2);amp=.05
    out=endpoint_metrics(u,u,u,Equation(.01,0),g,'discrete')
    assert out['prediction']['gradient_squared_mean']==pytest.approx((2*amp*n)**2)
    assert out['expected_mean_rate']==0
    assert out['prediction']['l2_energy_density']>0


@pytest.mark.parametrize('track',['discrete','continuum'])
def test_mean_balance_requires_real_derivative_and_reaction_is_not_conserved(track):
    from tdn.analysis.frontier.numerics import rhs
    u,g=field();eq=Equation(.01,2)
    derivative=rhs(u,eq,g,track)
    out=endpoint_metrics(u,u,u,eq,g,track,dict(accepted=True,time_derivative=derivative))
    assert out['expected_mean_rate']>0
    assert out['mean_balance_residual']<1e-13
    wrong=endpoint_metrics(u,u,u,eq,g,track,dict(accepted=True,time_derivative=derivative+.1))
    assert wrong['mean_balance_residual']==pytest.approx(.1)
    assert out['upper_rms'] is None # No invented reference uncertainty


def test_batch_mean_errors_do_not_cancel_and_bounds_are_counted():
    y=torch.full((2,1,8,8),.5,dtype=torch.float64);p=y.clone();p[0]+=.6;p[1]-=.6
    out=endpoint_metrics(p,y,y,Equation(0,1),Geometry((8,8),(1.,1.)),'discrete')
    assert abs(out['signed_mean_error'])<1e-15
    assert out['mean_error_rms']==pytest.approx(.6)
    assert out['physical_interval_violations']==128
    assert out['prediction']['below_zero_count']==64
    assert out['prediction']['above_one_count']==64
    assert out['centered_rms']<1e-15
    assert out['reference_accepted'] is False


def test_zero_target_relative_scale_explicit_and_nonfinite_prediction_retained():
    u,g=field();zero=torch.zeros_like(u)
    out=endpoint_metrics(zero,zero,zero,Equation(0,0),g,'discrete')
    assert out['relative_rms']==0 and out['relative_rms_floor_active']
    broken=u.clone();broken[0,0,0,0]=float('nan')
    bad=endpoint_metrics(broken,u,u,Equation(.01,1),g,'continuum')
    assert bad['finite'] is False and bad['error_rms']is None
    assert bad['nonfinite_prediction_values']==1
    json.dumps(bad,allow_nan=False)


def test_overflow_and_unresolved_modes_are_explicit_json_nulls():
    u,g=field(n=4);out=endpoint_metrics(u,u,u,Equation(0,1),g,'discrete')
    assert out['spectral_bands'][-1]['error_rms']is None
    huge=torch.full_like(u,1e200)
    bad=endpoint_metrics(huge,u,u,Equation(0,1),g,'discrete')
    assert not bad['metrics_finite']
    assert bad['metric_status']=='NONFINITE_DERIVED_METRICS'
    json.dumps(bad,allow_nan=False)


@pytest.mark.parametrize('reference',[{'uncertainty_rms':-1},{'uncertainty_rms':float('nan')},{'uncertainty_max_bound':float('inf')}])
def test_invalid_reference_bounds_rejected(reference):
    u,g=field()
    with pytest.raises(ValueError,match='uncertainty'):endpoint_metrics(u,u,u,Equation(0,1),g,'discrete',reference)


def test_invalid_shapes_targets_and_derivatives_rejected():
    u,g=field()
    with pytest.raises(ValueError,match='shapes'):endpoint_metrics(u[:,:,:8],u,u,Equation(0,1),g,'discrete')
    with pytest.raises(ValueError,match='finite'):endpoint_metrics(u,u*float('nan'),u,Equation(0,1),g,'discrete')
    with pytest.raises(ValueError,match='derivative'):endpoint_metrics(u,u,u,Equation(0,1),g,'discrete',{'time_derivative':torch.zeros(1)})


def test_cpu_paired_measurement_counts_scopes_and_no_invented_memory_or_tails():
    calls_seen={'a':0,'b':0}
    def call(name):
        calls_seen[name]+=1
        return torch.arange(32,dtype=torch.float64).sin().sum()
    answers,timing=measure_paired_metrics({name:(lambda name=name:call(name))for name in calls_seen},device='cpu',repeats=5,warmup=2,seed=74)
    assert calls_seen=={'a':8,'b':8}
    assert set(answers)=={'a','b'} and len(timing['rounds'])==5
    assert timing['memory_probe_seconds']==0 and timing['energy_joules']is None
    for method in timing['methods'].values():
        assert method['memory']['status']=='NA_CPU_ALLOCATOR_PEAK_UNAVAILABLE'
        assert method['memory']['absolute_peak_allocated_bytes'] is None
        assert method['tail_statistics']['p95_seconds'] is None
        assert method['tail_statistics']['p99_seconds'] is None
        assert method['tail_statistics']['observed_max_seconds']==max(method['samples_seconds'])
    json.dumps(timing,allow_nan=False)


def test_more_repeats_only_enable_empirical_not_certified_tail():
    _,timing=measure_paired_metrics({'x':lambda:1},repeats=100,warmup=1,memory_probe=False)
    tail=timing['methods']['x']['tail_statistics']
    assert tail['p95_seconds'] is not None and tail['p99_seconds']is None
    assert 'no tail-confidence' in tail['scope']


@pytest.mark.parametrize('track',['discrete','continuum'])
def test_logistic_rd_energy_gradient_matches_negative_rhs_and_translations(track):
    from tdn.analysis.frontier.numerics import rhs
    u,g=field(n=16,mode=2);u=u.clone().requires_grad_(True);eq=Equation(.003,2.)
    energy=energy_value(u,eq,g,track)
    gradient,=torch.autograd.grad(energy,u)
    direction=rhs(u,eq,g,track)
    derivative=(gradient*direction).sum()
    assert float(derivative.detach())==pytest.approx(-float(direction.detach().square().mean()),rel=2e-12,abs=1e-14)
    assert torch.max(torch.abs(gradient*u.numel()+direction))<2e-12
    shifted=torch.roll(u,(3,-2),(-2,-1))
    assert float(energy_value(shifted,eq,g,track).detach())==pytest.approx(float(energy.detach()),abs=1e-14)
    next_state=(u+1e-4*direction).detach()
    out=endpoint_metrics(next_state,next_state,u,eq,g,track)
    assert out['energy_change']<0 and out['energy_increase']==0
    assert out['energy_absolute_error']==0


@pytest.mark.parametrize('track',['discrete','continuum'])
@pytest.mark.parametrize('n',[8,9,16])
def test_energy_gradient_flow_with_full_random_nyquist_content(track,n):
    from tdn.analysis.frontier.numerics import rhs
    generator=torch.Generator().manual_seed(8)
    u=(.3+.3*torch.rand((1,1,n,n),generator=generator,dtype=torch.float64)).requires_grad_(True)
    eq=Equation(.001,2);geom=Geometry((n,n),(1.,1.))
    gradient,=torch.autograd.grad(energy_value(u,eq,geom,track),u)
    direction=rhs(u,eq,geom,track)
    assert float((gradient*direction).sum().detach())==pytest.approx(
        -float(physical_inner_product(direction,direction,geom,track).detach()),rel=2e-12,abs=2e-14)


def test_continuous_nyquist_inner_product_matches_resolved_cosine_integral():
    u,geom=field(n=16,mode=8);pure=u-.4
    assert float(physical_inner_product(pure,pure,geom,'discrete'))==pytest.approx(.05**2)
    assert float(physical_inner_product(pure,pure,geom,'continuum'))==pytest.approx(.05**2/2)
