"""Independent uncertainty and physical/deployment attribution of controls."""
import pytest
import torch
from tdn.analysis.response import numerics as n
from tdn.analysis.tangent_response.oracle import run_mechanism, independent_solve
from test_tangent_response_numerics import case


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_mechanism_control_coverage_uncertainties_and_bridge_are_distinct():
    a,_,eq,geom=case();u=a.float();actual=u.double()
    teacher,acquisition=independent_solve(actual,.04,eq,geom,'fd_same_grid',tolerance=1e-12)
    sample=dict(u0=u,teacher64=teacher,reference=dict(accepted=True,uncertainty_rms=1e-12,uncertainty_maximum=1e-12),
        record=dict(parent_id='fixed-parent',query_id='fixed-query',split='diagnostic',horizon=.04,target_kind='fd_same_grid',
                    grid=list(geom.grid),lengths=list(geom.lengths),kappa=eq.kappa,reaction_rate=eq.reaction_rate,
                    regime='mixed',role='primary',primary_case=True))
    result=run_mechanism([sample],dict(response_initial_steps=2,response_max_steps=8,response_tolerance=1e-6,
                                      source_nodes=4,max_source_nodes=8,path_tolerance=1e-6))
    row=result['rows'][0]
    assert set(row['controls'])=={'anchor','G','U','nonlinear1','nonlinear2','full_error_oracle','GL4','classical'}
    assert row['bridge_identity_maximum']<1e-16 and row['dense'] is False
    assert row['grid']==list(geom.grid) and row['kappa']==eq.kappa and row['reaction_rate']==eq.reaction_rate
    assert 'primary_status' in row and 'source_path_uncertainties' in row
    for metric in row['controls'].values():
        assert 'mean_mse' in metric and 'centered_mse' in metric and 'uncertainty_rms' in metric
        assert abs(metric['rms']**2-metric['mean_mse']-metric['centered_mse'])<1e-18
    tensors=result['private_tensors'][0]
    torch.testing.assert_close(tensors['deployment_defect'],tensors['physical_defect']+tensors['arithmetic_bridge'],atol=1e-16,rtol=1e-13)


def test_independent_galerkin_nested_products_are_not_assumed_associative():
    from tdn.analysis.tangent_response.oracle import NumpyEquation
    import numpy as np
    geom=n.Geometry((5,),(1.,));eq=n.Physics(.01,2.);equation=NumpyEquation(eq,geom,'galerkin_same_grid')
    x=np.arange(5)/5;a=np.cos(4*np.pi*x);b=a.copy();c=np.cos(2*np.pi*x)
    assert np.max(np.abs(equation.product(a,equation.product(b,c))-equation.product(equation.product(a,b),c)))>.1


@pytest.mark.parametrize('teacher,time_error,source_error,expected', [
    (1e-11, 2e-11, 3e-11, 'RESOLVED'),
    (3e-11, 4e-11, 4e-11, 'UNRESOLVED'),
    (float('nan'), 1e-11, 1e-11, 'UNRESOLVED'),
    (None, 1e-11, 1e-11, 'UNRESOLVED'),
    (1e-11, -1e-11, 1e-11, 'UNRESOLVED'),
])
def test_mechanism_combined_budget_censors_accepted_but_overbudget_components(monkeypatch,teacher,time_error,source_error,expected):
    """Status arithmetic fixture: fake estimates cannot establish oracle quality."""
    from tdn.analysis.tangent_response import oracle
    a,_,eq,geom=case();u=a.float()
    sample=dict(u0=u,teacher64=u.double(),eq=eq,geom=geom,
        reference=dict(accepted=True,uncertainty_rms=teacher,uncertainty_maximum=teacher),
        record=dict(parent_id='budget-fixture',split='diagnostic',horizon=.04,target_kind='fd_same_grid',
                    grid=list(geom.grid),lengths=list(geom.lengths),kappa=eq.kappa,reaction_rate=eq.reaction_rate))
    def metadata(value):
        return dict(accepted=True,uncertainty_rms=value,uncertainty_maximum=value,refinements=[dict(steps=4)])
    monkeypatch.setattr(oracle,'refined_forced',lambda path,*args,**kwargs:(torch.zeros_like(path.u0),metadata(time_error)))
    monkeypatch.setattr(oracle,'refined_source_quadrature',lambda path,*args,**kwargs:(torch.zeros_like(path.u0),metadata(source_error)))
    row=oracle.run_mechanism([sample],dict(total_uncertainty_budget=1e-10,path_tolerance=1e-10))['rows'][0]
    assert row['primary_status']==expected
    assert row['controls']['G']['status']==row['controls']['U']['status']==expected
    assert row['independent_path_uncertainty']['uncertainty_is_certificate'] is False
    if expected=='RESOLVED':
        assert row['controls']['U']['uncertainty_rms']<=1e-10
        assert row['controls']['U']['uncertainty_maximum']<=1e-10
    if teacher==3e-11:
        assert row['controls']['U']['uncertainty_rms']>1e-10
        assert row['response_uncertainties']['U']['accepted'] is True
        assert row['source_path_uncertainties']['U']['accepted'] is True


def test_nonfinite_independent_path_checks_censor_primary_controls(monkeypatch):
    import numpy as np
    from tdn.analysis.tangent_response import oracle
    a,_,eq,geom=case();u=a.float()
    sample=dict(u0=u,teacher64=u.double(),eq=eq,geom=geom,
        reference=dict(accepted=True,uncertainty_rms=1e-12,uncertainty_maximum=1e-12),
        record=dict(parent_id='path-fixture',split='diagnostic',horizon=.04,target_kind='fd_same_grid',
                    grid=list(geom.grid),lengths=list(geom.lengths),kappa=eq.kappa,reaction_rate=eq.reaction_rate))
    class CorruptedIndependentPath(oracle.NumpyEquation):
        def anchor_path(self,*args,**kwargs):
            row=super().anchor_path(*args,**kwargs)
            row['residual']=np.full_like(row['residual'],np.nan)
            return row
    monkeypatch.setattr(oracle,'NumpyEquation',CorruptedIndependentPath)
    metadata=dict(accepted=True,uncertainty_rms=1e-12,uncertainty_maximum=1e-12,refinements=[dict(steps=4)])
    monkeypatch.setattr(oracle,'refined_forced',lambda path,*args,**kwargs:(torch.zeros_like(path.u0),metadata))
    monkeypatch.setattr(oracle,'refined_source_quadrature',lambda path,*args,**kwargs:(torch.zeros_like(path.u0),metadata))
    row=oracle.run_mechanism([sample])['rows'][0]
    assert row['primary_status']=='UNRESOLVED'
    assert row['independent_path_uncertainty']['status']=='UNRESOLVED'
    assert row['controls']['U']['uncertainty_rms'] is None


@pytest.mark.parametrize('actual_source_accepted', [True,False])
def test_unused_interpolation_does_not_replace_actual_source_qualification(monkeypatch,actual_source_accepted):
    """Status fixture preserves a real source cap while ignoring an unused path."""
    from tdn.analysis.tangent_response import oracle
    a,_,eq,geom=case();u=a.float()
    sample=dict(u0=u,teacher64=u.double(),eq=eq,geom=geom,
        reference=dict(accepted=True,uncertainty_rms=1e-12,uncertainty_maximum=1e-12),
        record=dict(parent_id='quadrature-fixture',split='diagnostic',horizon=.04,target_kind='fd_same_grid',
                    grid=list(geom.grid),lengths=list(geom.lengths),kappa=eq.kappa,reaction_rate=eq.reaction_rate))
    def metadata(accepted,uncertainty):
        return dict(accepted=accepted,uncertainty_rms=uncertainty,uncertainty_maximum=uncertainty,
                    status='ACCEPTED' if accepted else 'UNRESOLVED',refinements=[dict(steps=4)])
    monkeypatch.setattr(oracle,'refined_forced',lambda path,*args,**kwargs:(torch.zeros_like(path.u0),metadata(True,1e-12)))
    monkeypatch.setattr(oracle,'refined_source_quadrature',lambda path,*args,**kwargs:
                        (torch.zeros_like(path.u0),metadata(actual_source_accepted,1e-12 if actual_source_accepted else None)))
    monkeypatch.setattr(oracle,'refined_path_response',lambda path,*args,**kwargs:(torch.zeros_like(path.u0),metadata(False,1.)))
    row=oracle.run_mechanism([sample],dict(path_interpolation_diagnostic=True))['rows'][0]
    assert row['primary_status']==('RESOLVED' if actual_source_accepted else 'UNRESOLVED')
    assert row['optional_path_interpolation_diagnostics']['U']['accepted'] is False
    if not actual_source_accepted:assert row['controls']['U']['uncertainty_rms'] is None


def test_accepted_gl_convergence_cannot_hide_large_direct_forced_disagreement(monkeypatch):
    from tdn.analysis.tangent_response import oracle
    a,_,eq,geom=case();u=a.float()
    sample=dict(u0=u,teacher64=u.double(),eq=eq,geom=geom,
        reference=dict(accepted=True,uncertainty_rms=1e-12,uncertainty_maximum=1e-12),
        record=dict(parent_id='crosscheck-fixture',split='diagnostic',horizon=.04,target_kind='fd_same_grid',
                    grid=list(geom.grid),lengths=list(geom.lengths),kappa=eq.kappa,reaction_rate=eq.reaction_rate,
                    panel='phase_intervention',phase_shift=.125,input_sha256='a'*64,active_frequency=[2,1],
                    cohort='fit',extrapolation=False))
    def metadata():
        return dict(accepted=True,uncertainty_rms=1e-12,uncertainty_maximum=1e-12,refinements=[dict(steps=4)])
    monkeypatch.setattr(oracle,'refined_forced',lambda path,*args,**kwargs:(torch.zeros_like(path.u0),metadata()))
    monkeypatch.setattr(oracle,'refined_source_quadrature',lambda path,*args,**kwargs:(torch.full_like(path.u0,2e-10),metadata()))
    row=oracle.run_mechanism([sample])['rows'][0]
    assert row['primary_status']=='UNRESOLVED'
    assert row['source_path_uncertainties']['U']['accepted'] is True
    assert row['source_path_uncertainties']['U']['direct_forced_crosscheck']['accepted'] is False
    assert row['controls']['U']['uncertainty_rms']>1e-10
    for key in ('cohort','panel','phase_shift','extrapolation','input_sha256','active_frequency'):
        assert row[key]==sample['record'][key]
