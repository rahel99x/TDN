"""Paired continuous phase interventions, not extra independent parent draws."""
import copy
import numpy as np
import pytest
import torch
from tdn.analysis.response.fields import make_parent,phase_variant,sample_parent,spectral_stress_parent
from tdn.analysis.response.numerics import inner,mean
from tdn.numerics.types import Geometry


@pytest.mark.parametrize('dimension,grid',[(1,(31,)),(2,(15,15)),(2,(16,16))])
def test_phase_twin_has_equal_power_but_changes_translation_invariant_triad(dimension,grid):
    parent=make_parent(88301,'validation',5,dimension=dimension)
    original=phase_variant(parent,'original');twin=phase_variant(parent,'phase_twin')
    assert original['parent_id']==twin['parent_id']==parent['parent_id']
    assert original['parent_sha256']==twin['parent_sha256']==parent['parent_sha256']
    assert original['descendant_sha256']!=twin['descendant_sha256']
    a=sample_parent(original,grid);b=sample_parent(twin,grid)
    axes=tuple(range(2,a.ndim));fa=torch.fft.fftn(a,dim=axes,norm='forward');fb=torch.fft.fftn(b,dim=axes,norm='forward')
    torch.testing.assert_close(fa.abs().square(),fb.abs().square(),rtol=1e-12,atol=1e-15)
    torch.testing.assert_close(mean(a),mean(b),rtol=1e-14,atol=1e-15)
    # The bispectrum is invariant under translation but flips sign here.
    if dimension==2:
        t1=fa[0,0,1,0]*fa[0,0,0,1]*fa[0,0,1,1].conj()
        t2=fb[0,0,1,0]*fb[0,0,0,1]*fb[0,0,1,1].conj()
    else:
        t1=fa[0,0,1]*fa[0,0,2]*fa[0,0,3].conj()
        t2=fb[0,0,1]*fb[0,0,2]*fb[0,0,3].conj()
    assert abs(t1)>1e-14
    torch.testing.assert_close(t1,-t2,rtol=1e-11,atol=1e-15)
    assert float((a-b).abs().max())>1e-5
    assert min(float(a.min()),float(b.min()))>=0 and max(float(a.max()),float(b.max()))<=1


def test_single_mode_phase_twin_is_labeled_translation_control():
    parent=make_parent(881,'validation',0,regime='isolated',dimension=1)
    a=sample_parent(phase_variant(parent,'original'),(32,))
    twin=phase_variant(parent,'phase_twin');b=sample_parent(twin,(32,))
    assert 'translation-only' in twin['phase_scope']
    torch.testing.assert_close(torch.roll(a,16,dims=-1),b,rtol=1e-13,atol=1e-14)


@pytest.mark.parametrize('kind',['grid_relative','broadband'])
def test_spectral_stress_stays_bounded_and_fixed_during_teacher_refinement(kind):
    parent=make_parent(880,'validation',2,regime='smooth',dimension=2)
    before=copy.deepcopy(parent);stress=spectral_stress_parent(parent,32,kind)
    assert parent==before
    assert stress['stress_parent_id']==parent['parent_id']
    assert 'not an additional independent parent' in stress['independence_scope']
    assert max(abs(k) for term in stress['terms'] for k in term['mode'])>3
    coarse=sample_parent(stress,(32,32));fine=sample_parent(stress,(128,128))
    torch.testing.assert_close(coarse,fine[:,:,::4,::4],rtol=1e-13,atol=1e-14)
    assert float(fine.min())>=0 and float(fine.max())<=1
    geom=Geometry((32,32),(1.,1.))
    assert float(inner(coarse-mean(coarse),coarse-mean(coarse),geom,'galerkin_same_grid'))>0
    another=spectral_stress_parent(parent,64,kind)
    assert another['terms']!=stress['terms']
