"""Mathematical and provenance checks for finite rough-field parents."""
import json
import math

import numpy as np
import pytest
import torch

from tdn.analysis.adjacent.fields import (
    ALPHAS, FieldParent, make_parent, phase_variant, roughness_metrics,
    sample_field, translate_parent, with_bandwidth,
)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("generator", ["ridge", "multiscale_2d"])
def test_all_roughness_labels_mean_rms_and_analytic_range(alpha, generator):
    parent = make_parent(702, alpha, generator=generator, levels=3)
    u = sample_field(parent, 32)
    assert float(u.mean()) == pytest.approx(.43, abs=3e-15)
    assert float((u-u.mean()).square().mean().sqrt()) == pytest.approx(.07, abs=3e-15)
    lo, hi = parent.continuous_range_bound
    assert 0 <= lo <= float(u.min()) <= float(u.max()) <= hi <= 1
    if alpha == 1:
        assert "finite-bandwidth H=0" in parent.dimension_label
        assert parent.hurst == 0
    else:
        assert parent.hurst > 0
    assert "finite sample" in parent.dimension_label or alpha == 1


def test_canonical_ridge_is_finite_weierstrass_not_a_fractal_sample():
    parent = make_parent(1, .4, generator="ridge", levels=5, phase_mode="structured")
    assert parent.modes == ((1,0),(2,0),(4,0),(8,0),(16,0))
    assert np.asarray(parent.amplitudes[1:])/parent.amplitudes[:-1] == pytest.approx(2**(-.6))
    assert "ideal canonical ridge" in parent.dimension_label
    u = sample_field(parent, 64)[0,0]
    assert torch.equal(u[:,0], u[:,33])
    # Canonical structured ridges with another seed are the same field and
    # cannot inflate independent parent counts.
    other = make_parent(7, .4, generator="ridge", levels=5, phase_mode="structured")
    assert other.cluster_id == parent.cluster_id
    assert torch.equal(sample_field(other,64), sample_field(parent,64))


def test_multiscale_is_truly_two_dimensional_and_independent():
    a = make_parent(715, .5, levels=3)
    b = make_parent(716, .5, levels=3)
    for shell in range(3):
        modes = [m for m,j in zip(a.modes,a.shell_indices) if j == shell]
        assert np.linalg.matrix_rank(modes) == 2
    assert a.cluster_id != b.cluster_id
    assert not torch.allclose(sample_field(a,32), sample_field(b,32))
    assert "theorem not transferred" in a.dimension_label


def test_grid_refinement_preserves_continuous_parent_and_all_coefficients():
    parent = make_parent(34, .9, levels=3)
    coarse = sample_field(parent, (16,24), domain=(2.,3.))
    fine = sample_field(parent, (32,48), domain=(2.,3.))
    assert torch.allclose(fine[...,::2,::2],coarse,atol=2e-15,rtol=0)
    # Fourier restriction, not just nodal subsampling: amplitudes at physical
    # signed modes agree across the paired grids.
    a,b = [torch.fft.fftn(u[0,0],norm="forward") for u in (coarse,fine)]
    for k,l in parent.modes:
        assert abs(complex(a[k,l])-complex(b[k,l])) < 2e-15
    assert parent.workload == "fixed_physical_bandwidth"
    with pytest.raises(ValueError,match="Nyquist"):
        sample_field(parent,8)


def test_bandwidth_change_is_a_changed_workload_not_grid_refinement():
    parent = make_parent(98,.7,levels=2)
    changed = with_bandwidth(parent,3)
    assert changed.cluster_id == parent.cluster_id
    assert changed.modes[:len(parent.modes)] == parent.modes
    assert changed.phases[:len(parent.phases)] == parent.phases
    assert changed.rms == parent.rms
    assert changed.workload == "changed_physical_bandwidth_at_fixed_total_RMS"
    assert not torch.allclose(sample_field(parent,32),sample_field(changed,32))


def test_phase_controls_match_power_but_are_not_new_independent_fields():
    parent = make_parent(901,.8,levels=3)
    structured = phase_variant(parent,"structured")
    assert structured.cluster_id == parent.cluster_id
    a,b = [torch.fft.fftn(sample_field(p,32),dim=(-2,-1),norm="forward") for p in (parent,structured)]
    assert torch.allclose(a.abs().square(),b.abs().square(),atol=2e-17,rtol=1e-12)
    assert not torch.allclose(a,b)


def test_translation_is_exact_paired_symmetry_and_retains_power():
    parent = make_parent(31,.5,levels=3)
    translated = translate_parent(parent,(3/32,-2/32))
    u = sample_field(parent,32)
    shifted = sample_field(translated,32)
    assert torch.allclose(shifted,torch.roll(u,(-3,2),dims=(-2,-1)),atol=3e-15,rtol=0)
    assert translated.cluster_id == parent.cluster_id
    assert translated.identity_sha256 != parent.identity_sha256


def test_serialized_coefficients_and_hash_are_validated():
    parent = make_parent(7,.3)
    data = json.loads(json.dumps(parent.to_dict()))
    restored = FieldParent.from_dict(data)
    assert parent == restored
    assert torch.equal(sample_field(data,8),sample_field(parent,8))
    data["phases"][0] += .1
    with pytest.raises(ValueError,match="hash mismatch"):
        FieldParent.from_dict(data)


def test_no_clipping_when_mean_rms_and_range_cannot_all_be_matched():
    with pytest.raises(ValueError,match="cannot all be matched"):
        make_parent(8,.8,mean=.03,rms=.1,levels=3)
    parent = make_parent(8,.8,mean=.03,rms=.002,levels=3)
    assert float((sample_field(parent,32)-.03).square().mean().sqrt()) == pytest.approx(.002)


def test_physical_spectral_moments_and_shell_parseval():
    parent = make_parent(44,.6,generator="ridge",levels=3)
    u = sample_field(parent,128,domain=(2.,1.))
    result = roughness_metrics(u,domain=(2.,1.))
    exact_gradient = sum((a*a/2)*(2*math.pi*k[0]/2)**2 for a,k in zip(parent.amplitudes,parent.modes))
    assert result["gradient_energy"] == pytest.approx(exact_gradient,rel=1e-12)
    assert result["spectral_variance"] == pytest.approx(parent.rms**2,rel=1e-12)
    assert sum(s["energy"] for s in result["shell_energies"]) == pytest.approx(parent.rms**2,rel=1e-12)
    assert result["roughness_estimates"][0]["points"] == 5
    assert result["roughness_estimates"][1]["empirical_H"] is None
    assert result["dimension_inference"].startswith("NA")


def test_no_roughness_fit_on_two_scales_or_constant_field():
    result = roughness_metrics(sample_field(make_parent(78,.5),16))
    assert all(e["empirical_H"] is None for e in result["roughness_estimates"])
    result = roughness_metrics(torch.ones(1,1,64,64)*.4)
    assert result["mean_squared_angular_wavenumber"] is None
    assert all(e["empirical_H"] is None for e in result["roughness_estimates"])


@pytest.mark.parametrize("kwargs", [dict(alpha=0),dict(alpha=1.1),dict(alpha=float("nan")),
                                     dict(levels=0),dict(orientation=(0,0)),dict(seed=-1),
                                     dict(generator="third_dimension"),dict(rms=-.1)])
def test_invalid_fields_rejected(kwargs):
    with pytest.raises(ValueError):
        make_parent(**(dict(seed=2,alpha=.5)|kwargs))


def test_large_bandwidth_parent_uses_sparse_storage():
    parent = make_parent(18,.5,levels=12,rms=.01)
    assert len(parent.modes) <= 48
    assert max(max(abs(k) for k in m) for m in parent.modes) > 1000
