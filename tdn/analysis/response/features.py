"""Train-frozen scales and horizon-independent, phase-aware physical inputs.

No target, endpoint or teacher enters this encoder. Products are formed on the
declared working grid before learned spectral compression. The power ablation
intentionally retains only translation-invariant scalar information: its
learned centered output cannot reconstruct the missing phase.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from . import numerics as n
from .contracts import Physics

PHYSICAL_BAND_EDGES = (2., 4., 8.)  # cycles per physical unit, not grid fractions
FEATURE_NAMES = ('state_scaled','centered_state','heat_quarter','heat_full','quadratic','cubic',
                 'mean','variance','log_stiffness','log_reaction','filtered_variance_ratio',
                 'gradient_ratio','third_moment','fourth_moment','minimum_scaled','maximum_scaled',
                 'band_0_2','band_2_4','band_4_8','band_above_8',
                 'band_energy_0_2','band_energy_2_4','band_energy_4_8','band_energy_above_8')
FEATURE_CHANNELS = len(FEATURE_NAMES)


def physical_band_fields(v, geom):
    """Radial physical-frequency masks, including exact boundary modes below.

    Frequencies have units inverse geometry-length; upper resolved frequency
    changes with sampling, but the declared physical cutoffs never do. The DC
    field is omitted and Nyquist synthesis follows the same full-FFT convention.
    """
    radial2 = torch.zeros(geom.grid, dtype=v.dtype, device=v.device)
    for i,(size,dx) in enumerate(zip(geom.grid,geom.dx)):
        freq=torch.fft.fftfreq(size,d=dx,dtype=v.dtype,device=v.device)
        shape=[1]*geom.ndim;shape[i]=size
        radial2=radial2+freq.reshape(shape).square()
    spectrum=torch.fft.fftn(v,dim=n.axes(v),norm='forward')
    masks=[];lower=0.
    for upper in (*PHYSICAL_BAND_EDGES,float('inf')):
        masks.append((radial2>lower**2)&(radial2<=upper**2))
        lower=upper
    return [n.center(torch.fft.ifftn(spectrum*mask,dim=n.axes(v),norm='forward').real) for mask in masks]



class FrozenNormalization(nn.Module):
    """Persistent, independently fitted mean/centered/full-output scales."""
    def __init__(self, values=None):
        super().__init__()
        values = dict(values or {})
        floor = float(values.get('floor', 1e-7))
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError('Normalization floor must be finite and positive')
        for key, default in [('mean_scale', 1.), ('centered_scale', 1.),
                             ('full_mean', 0.), ('full_scale', 1.), ('floor', floor)]:
            value = float(values.get(key, default))
            if not math.isfinite(value) or (key != 'full_mean' and value <= 0):
                raise ValueError(f'Invalid frozen normalization {key}')
            # Scales below the declared arithmetic floor cannot masquerade as
            # informative residual units. The adjusted values are serialized.
            if key in ('mean_scale', 'centered_scale', 'full_scale'):
                value = max(value, floor)
            self.register_buffer(key, torch.tensor(value, dtype=torch.float64))

    def metadata(self):
        return {key: float(getattr(self, key).detach().cpu()) for key in
                ('mean_scale', 'centered_scale', 'full_mean', 'full_scale', 'floor')}


def build_features(u, eq, geom, target, normalization, *, t_ref=1., mode='phase'):
    """Return documented physical channels and useful per-parent, horizon-independent scalars."""
    n.validate(u, 0., geom, target)
    if mode not in ('phase', 'power') or not math.isfinite(t_ref) or t_ref <= 0:
        raise ValueError('Feature mode phase/power and positive reference time required')
    m, v = n.mean(u), n.center(u)
    variance = n.inner(v, v, geom, target)
    scale = normalization.full_scale.to(u)
    floor = normalization.floor.to(u)
    safe_var = variance + floor.square()
    # No normalization by a future target or tiny per-example baseline defect.
    amplitude = torch.sqrt(safe_var)
    heat_quarter = n.heat(v, .25*t_ref, eq, geom, target)
    heat_full = n.heat(v, t_ref, eq, geom, target)
    square = n.product(v, v, geom, target)
    cube = n.product(v, square, geom, target)  # sequential projected cubic
    unit = Physics(1., 0.)
    gradient_energy = -n.inner(v, n.generator(v, unit, geom, target), geom, target)
    stiffness = n.scalar(eq.kappa, u)*t_ref*gradient_energy/safe_var
    reaction = n.scalar(eq.reaction_rate, u)*t_ref
    filtered_variance = n.inner(heat_full, heat_full, geom, target)
    broad = lambda x: x.expand_as(u)
    scalar_features = [m, variance/scale.square(),
                       torch.log1p(stiffness), torch.log1p(reaction),
                       filtered_variance/safe_var,
                       gradient_energy/(safe_var+gradient_energy),
                       n.inner(v, square, geom, target)/(amplitude**3),
                       n.inner(square, square, geom, target)/(safe_var**2)]
    if mode == 'phase':
        spatial = [(u-normalization.full_mean.to(u))/scale, v/scale,
                   heat_quarter/scale, heat_full/scale,
                   n.center(square)/scale.square(), n.center(cube)/scale**3]
    else:
        # All six replacements are moments/energies, not a phase-leaking raw
        # field, extrema map, coordinate channel, or teacher diagnostic.
        fourth_spectral_moment = n.inner(n.generator(v, unit, geom, target),
            n.generator(v, unit, geom, target), geom, target)
        power_scalars = [m, variance/scale.square(),
                   n.inner(heat_quarter, heat_quarter, geom, target)/safe_var,
                   filtered_variance/safe_var,
                   gradient_energy/(safe_var+gradient_energy),
                   fourth_spectral_moment/(safe_var+fourth_spectral_moment)]
        spatial = [broad(x) for x in power_scalars]
        # Cubic and quartic pointwise moments carry phase relationships and
        # must not leak into this deliberately power-only negative control.
        scalar_features[-2:] = power_scalars[-2:]
    bands = physical_band_fields(v, geom)
    energies = [n.inner(b,b,geom,target)/scale.square() for b in bands]
    if mode == 'phase':
        extrema = [(u.amin(n.axes(u),keepdim=True)-normalization.full_mean.to(u))/scale,
                   (u.amax(n.axes(u),keepdim=True)-normalization.full_mean.to(u))/scale]
        band_channels = [b/scale for b in bands]
    else:
        # Extrema contain relative-phase information. Power-only explicitly
        # replaces them with m +/- RMS and band fields with RMS broadcasts.
        extrema = [(m-amplitude-normalization.full_mean.to(u))/scale,
                   (m+amplitude-normalization.full_mean.to(u))/scale]
        band_channels = [broad(torch.sqrt(e+floor.square()/scale.square())) for e in energies]
    features = torch.cat(spatial+[broad(x) for x in scalar_features+extrema]+band_channels+[broad(e) for e in energies], dim=1)
    return features, dict(mean=m, variance=variance, stiffness=stiffness,
                          reaction=reaction, gradient_energy=gradient_energy)


def correction_gate(stats, target):
    """Known physical nulls, without asserting a false Galerkin kappa=0 null.

    For same-grid FD, exact reaction and heat subflows justify zero splitting
    corrections for constant states and kappa=0. Finite projected-reaction RK4
    has a numerical residual even on constants; neither that remainder nor
    unresolved closure justifies those nulls in the Galerkin tracks.
    """
    r, stiffness, variance = stats['reaction'], stats['stiffness'], stats['variance']
    gate = r/(1+r)
    if target == 'fd_same_grid':
        gate = gate * (variance/(variance+1e-6)) * (stiffness/(1+stiffness))
    return gate
