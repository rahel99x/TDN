"""Mathematical identifiability and executed numerical diagnostic regressions."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tdn.analysis.adjacent.diagnostics import (
    _Experiment, _model, _state, acceptable_gain_interval, basis_oracle,
    run_diagnostic, scalar_oracle,
)
from tdn.analysis.frontier import numerics
from tdn.analysis.portfolio.models import physical_features
from tdn.numerics import Equation, Geometry


def test_scalar_projection_is_best_l2_and_bounds_are_actual_constraints():
    q=np.array([1.,-2.,.3]);d=3*q+np.array([2.,1.,0.])
    result=scalar_oracle(q,d)
    assert result['coefficient']==pytest.approx(3.)
    assert result['error_rms']<scalar_oracle(q,d,(.25,1.75))['error_rms']
    assert scalar_oracle(q,d,(.25,1.75))['coefficient']==1.75
    residual=result['coefficient']*q-d
    assert abs(q@residual)<1e-13


@pytest.mark.parametrize('q,d',[(np.zeros(3),np.ones(3)),(np.ones(3),np.zeros(3)),
                              (np.full(3,1e-13),np.ones(3)),(np.ones(3),np.full(3,1e-13))])
def test_oracles_do_not_infer_gain_from_uncertainty_sized_signals(q,d):
    assert not scalar_oracle(q,d,uncertainty=1e-12)['resolved']
    assert not basis_oracle(q[None],d,uncertainty=1e-12)['resolved']


def test_projection_does_not_claim_to_optimize_combined_peak_objective():
    basis=np.array([[1.,1.,1.]])
    d=np.array([0.,0.,9.])
    result=basis_oracle(basis,d)
    assert result['coefficients']==pytest.approx([3.])
    assert result['objective_optimized']=='squared_L2_only'
    combined_at_l2=result['combined_objective']
    alternative=3.5*basis[0]-d
    assert np.mean(alternative**2)+.1*np.max(alternative**2)<combined_at_l2


def test_rank_deficient_basis_reports_nonidentifiable_coefficients():
    q=np.linspace(-1,1,15)
    result=basis_oracle(np.stack((q,2*q)),3*q,reference_difference=.01*q)
    assert result['effective_rank']==1
    assert result['rank_deficient']
    assert not result['coefficient_values_identifiable']
    assert result['condition_number'] is None
    assert result['error_rms']<1e-14


def test_algebraically_full_rank_can_be_unidentifiable_at_reference_floor():
    basis=np.array([[1.,0.,0.],[0.,1e-10,0.]])
    result=basis_oracle(basis,np.array([1.,1e-10,0.]),uncertainty=1e-8)
    assert result['effective_rank']==2
    assert not result['coefficient_values_identifiable']
    assert result['coefficient_uncertainty_radius_l2']>100


def test_bounded_multichannel_fit_and_reference_sensitivity():
    basis=np.eye(3)
    result=basis_oracle(basis,np.array([2.,.1,1.]),bounds=(.25,1.75),
                        reference_difference=np.array([.001,0.,0.]))
    assert result['coefficients']==pytest.approx([1.75,.25,1.],abs=1e-10)
    assert result['coefficient_refinement_sensitivity_l2']==pytest.approx(.001)
    assert result['optimizer']['success']


def test_acceptable_gain_intervals_handle_flat_empty_and_overlapping_objectives():
    assert acceptable_gain_interval(np.zeros(3),np.zeros(3),1e-6)==[.25,1.75]
    assert acceptable_gain_interval(np.zeros(3),np.ones(3),.1) is None
    interval=acceptable_gain_interval(np.ones(3),np.ones(3),.1)
    assert interval==pytest.approx([.9,1.1])
    for gain in np.linspace(.25,1.75,100):
        permitted=np.sqrt(np.mean((gain*np.ones(3)-1)**2))<=.1
        assert permitted==(interval[0]<=gain<=interval[1])


@pytest.mark.parametrize('track',['discrete','continuum'])
def test_feature_collision_fields_match_actual_three_features(track):
    geo=Geometry((8,8),(1.,1.));eq=Equation(.004,3.)
    a=_state(8,'mixed',phase=0.,amplitude=.11)
    b=_state(8,'mixed',phase=np.pi/2,amplitude=.11)
    features=[physical_features(u,numerics.validate(u,.2,geo,track),eq,geo,track) for u in (a,b)]
    assert torch.max(torch.abs(features[0]-features[1]))<1e-13
    assert torch.max(torch.abs(a-b))>.01


def test_grid_refinement_can_hold_physical_cutoff_fixed():
    coarse=_model('quad2_fixed','continuum',8,modes=2)
    fine=_model('quad2_fixed','continuum',16,modes=2)
    assert coarse.modes==fine.modes==2
    assert _model('quad2_fixed','continuum',16).modes==4


@pytest.fixture(scope='module')
def executed(tmp_path_factory):
    root=tmp_path_factory.mktemp('adjacent-diagnostics')
    old=torch.get_num_threads();torch.set_num_threads(1)
    results={}
    try:
        for program in ('D01','D02','D04','D05','D06','D08'):
            directory=root/program
            summary=run_diagnostic(program,'smoke',directory)
            results[program]=(summary,json.loads((directory/'diagnostic_rows.json').read_text()),directory)
    finally:
        torch.set_num_threads(old)
    return results


@pytest.mark.parametrize('program',['D01','D02','D04','D05','D06','D08'])
def test_bounded_diagnostic_retains_actual_data_and_target(program,executed):
    summary,rows,path=executed[program]
    assert summary['status']=='COMPLETED'
    assert summary['unresolved_references']==0
    assert rows and all(r['diagnostic']==program for r in rows)
    assert {r['track'] for r in rows}=={'discrete','continuum'}
    assert len({r['case_id'] for r in rows})==len(rows)
    with np.load(path/'arrays.npz',allow_pickle=False) as data:
        for row in rows:
            if 'array_prefix' in row:
                assert any(k.startswith(row['array_prefix']+'__') for k in data.files)


def test_d01_unresolved_and_initializer_evidence_is_labeled(executed):
    _,rows,_=executed['D01']
    oracles=[r for r in rows if r['category']=='oracle']
    assert any(not r['oracle']['resolved'] for r in oracles if r['regime']=='nearly_constant')
    assert all(r['relative_backbone_improvement'] is None for r in oracles if not r['oracle']['resolved'])
    initial=[r for r in rows if r['method']=='quad2_conditioned']
    assert all(r['checkpoint_status']=='fresh_initialized_no_trained_checkpoint' for r in initial)
    assert {r['basis'] for r in oracles}=={'scalar','output_band','interaction_channel','quadratic_cubic'}


def test_d02_claim_scope_preserves_node_search_caveat(executed):
    _,rows,_=executed['D02']
    pairs=[r for r in rows if r['category']=='feature_collision']
    assert pairs and all(r['features_matched'] for r in pairs)
    assert all(r['arbitrary_node_conditioner_insufficiency'] is None for r in pairs)
    assert all(r['finite_node_search_not_global_node_optimum'] for r in pairs)
    assert all(r['reference_accepted'] for r in pairs if r['response_insufficiency_demonstrated'])


def test_d04_signed_phase_and_near_zero_floor_preserved(executed):
    _,rows,_=executed['D04']
    assert {'phase0','phase90','phase180','separation','cancellation'}<={r['sweep']['label'] for r in rows}
    for row in rows:
        expected=complex(row['coefficient_reference']['real'],row['coefficient_reference']['imag'])
        actual=complex(row['coefficient_predicted']['real'],row['coefficient_predicted']['imag'])
        assert row['absolute_coefficient_error']==pytest.approx(abs(actual-expected))
        if abs(expected)<=row['coefficient_scale_floor']:
            assert row['phase_error'] is None
    fno=[r for r in rows if r['method']=='fno_small']
    assert all(r['local_fno_path_intact'] and not r['architecture_inferiority_claim_permitted'] for r in fno)


def test_d05_quadrature_order_and_backbone_contamination_are_separate(executed):
    _,rows,_=executed['D05']
    quadrature=[r for r in rows if r['category']=='quadrature_amplitude']
    assert {r['nodes'] for r in quadrature}=={2,4,16,32}
    assert all(r['quadrature_reference_nodes']==32 for r in quadrature)
    parity=[r for r in rows if r['category']=='amplitude_parity']
    assert all('background_integrator_error_rms' in r and 'linear_integrator_error_rms' in r for r in parity)
    assert any(r['background_integrator_error_rms']>1e-12 for r in parity if r['track']=='continuum')
    assert all(r['reference_accepted'] for r in rows if r['category']=='observed_order' and r['slope'] is not None)


def test_d06_autonomous_and_teacher_forced_are_distinct_with_correct_mean(executed):
    _,rows,_=executed['D06']
    trajectories=[r for r in rows if r['category']=='trajectory']
    for row in trajectories:
        assert len(row['times'])==len(row['trajectory_error_rms'])==len(row['schedule'])+1
        assert len(row['teacher_forced_one_step_rms'])==len(row['schedule'])
        mean=np.asarray(row['reference_observables']['mean'])
        variance=np.asarray(row['reference_observables']['variance'])
        expected=3*(mean-mean**2-variance)
        assert np.asarray(row['reference_observables']['mean_rhs'])==pytest.approx(expected,abs=3e-14)
    summaries=[r for r in rows if r['category']=='rollout_summary']
    assert all(r['worst_rms']>=r['error_rms'] for r in summaries)
    assert all(r['mean_conservation_not_assumed'] for r in summaries)


def test_failure_and_completed_output_preservation(tmp_path):
    def stop():
        raise TimeoutError('bounded test interrupt')
    path=tmp_path/'interrupted'
    with pytest.raises(TimeoutError):
        run_diagnostic('D01','smoke',path,check_budget=stop)
    summary=json.loads((path/'diagnostic_summary.json').read_text())
    assert summary['status']=='FAILED'
    assert (path/'arrays.npz').exists()
    with pytest.raises(FileExistsError):
        run_diagnostic('D01','smoke',path)


def test_reference_steps_honor_frozen_protocol_cap(tmp_path):
    ex=_Experiment('D06','development',tmp_path,{'diagnostics':{'max_reference_substeps':16}},lambda:None)
    u=_state(8,'low');geo=Geometry((8,8),(1.,1.));eq=Equation(.004,3.)
    _,record,_=ex.reference(u,3.,eq,geo,'discrete',parent_id='cap-test')
    assert max(record['refinement_counts'])<=16
    assert not record['reference_accepted']
