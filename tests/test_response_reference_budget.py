"""Component allocation retains the combined projected-teacher acceptance."""
import math

import pytest
import torch

from tdn.analysis.response import references
from tdn.analysis.response.contracts import Physics
from tdn.numerics.types import Geometry


def _options():
    return dict(tolerance=1e-10,spatial_factors=[2,4,8],max_grid=32,
        projected_error_budget=dict(temporal_tolerances=[2e-11]*3,spatial_tolerance=6e-11))


def test_component_allocation_validates_every_adjacent_pair_and_preserves_default():
    assert references.projected_error_budget({'tolerance':1e-10}) is None
    allocated=references.projected_error_budget(_options())
    assert allocated['adjacent_combined_tolerances']==pytest.approx([1e-10,1e-10])
    changed=_options();changed['projected_error_budget']['temporal_tolerances'][-1]=3e-11
    with pytest.raises(ValueError,match='sum'):
        references.projected_error_budget(changed)


@pytest.mark.parametrize('bad',[0.,-1.,math.nan,math.inf,True,None,'not-a-number'])
@pytest.mark.parametrize('component',['temporal','spatial','total'])
def test_component_allocation_rejects_nonpositive_nonfinite_budgets(bad,component):
    options=_options()
    if component=='temporal':options['projected_error_budget']['temporal_tolerances'][1]=bad
    elif component=='spatial':options['projected_error_budget']['spatial_tolerance']=bad
    else:options['tolerance']=bad
    with pytest.raises(ValueError):references.projected_error_budget(options)


def test_component_budget_binds_grid_count_and_order():
    options=_options();options['projected_error_budget']['temporal_tolerances'].pop()
    with pytest.raises(ValueError,match='match'):references.projected_error_budget(options)
    options=_options();options['spatial_factors']=[2,8,4]
    with pytest.raises(ValueError,match='increasing'):references.projected_error_budget(options)


def test_projected_solver_uses_each_exact_tolerance_and_records_grid_cap(monkeypatch):
    observed=[]
    def temporal(initial,h,equation,geometry,target,options,budget):
        observed.append(options['tolerance'])
        value=initial+len(observed)*8e-11
        return value,dict(accepted=True,status='ACCEPTED',uncertainty_rms=1e-11,uncertainty_maximum=1e-11)
    monkeypatch.setattr(references,'temporal_reference',temporal)
    options=_options();options['max_grid']=16
    _,meta=references.generate_reference(torch.full((1,1,4),.3),.1,Physics(.003,3.),
        Geometry((4,),(1.,)),'galerkin_projected_fine',options)
    assert observed==[2e-11,2e-11]
    assert not meta['accepted']  # Spatial component exceeds its own 6e-11 allocation.
    assert len(meta['resolution_attempts'])==3
    assert meta['resolution_attempts'][-1]['status']=='GRID_CAP_OR_INVALID_FACTOR'
    assert meta['uncertainty_maximum']>options['tolerance']-1e-16


def test_individually_accepted_temporal_components_do_not_relax_combined_acceptance(monkeypatch):
    observed=[]
    def temporal(initial,h,equation,geometry,target,options,budget):
        observed.append(options['tolerance'])
        return initial+len(observed)*5e-11,dict(accepted=True,status='ACCEPTED',
            uncertainty_rms=3e-11,uncertainty_maximum=3e-11)
    monkeypatch.setattr(references,'temporal_reference',temporal)
    _,meta=references.generate_reference(torch.full((1,1,4),.3),.1,Physics(.003,3.),
        Geometry((4,),(1.,)),'galerkin_projected_fine',_options())
    assert len(observed)==3 and not meta['accepted']
    assert meta['uncertainty_rms']>1e-10
    assert all(r['attempted'] for r in meta['resolution_attempts'])


def test_each_component_budget_is_enforced_even_when_combined_estimate_fits(monkeypatch):
    calls=[]
    def temporal(initial,h,equation,geometry,target,options,budget):
        calls.append(geometry.grid)
        return initial+len(calls)*1e-11,dict(accepted=True,status='ACCEPTED',
            uncertainty_rms=4e-11,uncertainty_maximum=4e-11)
    monkeypatch.setattr(references,'temporal_reference',temporal)
    _,meta=references.generate_reference(torch.full((1,1,4),.3),.1,Physics(.003,3.),
        Geometry((4,),(1.,)),'galerkin_projected_fine',_options())
    assert meta['uncertainty_rms']<1e-10
    assert not meta['accepted']
    assert not meta['resolution_attempts'][-1]['temporal_pair_budget_satisfied']


def test_real_projected_constant_solve_allocates_margin_and_retains_unused_factors():
    options=_options();options.update(initial_steps=4,max_steps=256,max_seconds_per_query=5)
    initial=torch.full((1,1,4),.3)
    value,meta=references.generate_reference(initial,.02,Physics(.003,.5),Geometry((4,),(1.,)),
        'galerkin_projected_fine',options)
    exact=initial.double()/(initial.double()+(1-initial.double())*math.exp(-.5*.02))
    assert meta['accepted']
    torch.testing.assert_close(value,exact,rtol=0,atol=1e-10)
    assert meta['uncertainty_rms']<=1e-10 and meta['uncertainty_maximum']<=1e-10
    assert [x['temporal_tolerance'] for x in meta['resolution_attempts'] if x['attempted']]==[2e-11,2e-11]
    assert meta['resolution_attempts'][-1]['status']=='NOT_NEEDED_AFTER_ACCEPTED_PAIR'
