"""Inspectable endpoint, temporal, and coupled finite-time response models.

The coupled model approximates the full mean trajectory and centered source;
its quadrature implements the declared mean-dependent variation-of-constants
formula, not an exact nonlinear solution. Every arithmetic path charges its
anchor and mean-path work. No teacher is available to an inference method.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from tdn.models.temporal import TemporalMLP
from tdn.research.premix_neural import SymmetricSpectralMixer
from tdn.analysis.roadmap.numerics import quadrature
from . import numerics as n
from .features import FrozenNormalization, FEATURE_CHANNELS, FEATURE_NAMES, PHYSICAL_BAND_EDGES, build_features, correction_gate
from .trajectory import source_response, homogeneous_residual_response, mean_propagator
from .kernels import background
from .temporal_basis import endpoint_basis, target_time_order, correction_gains

ALIASES = {'endpoint': 'residual_fno', 'endpoint_tdn': 'residual_fno',
           'original_tdn': 'temporal'}
FAMILIES = ('direct_fno', 'residual_fno', 'temporal', 'coupled_source', 'linear_response')
DIAGNOSTIC_FAMILIES = ('adaptive_temporal', 'spatial_temporal')
ALL_FAMILIES = FAMILIES+DIAGNOSTIC_FAMILIES


class Spectral1D(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight_real = nn.Parameter(torch.randn(width, width, modes+1)/math.sqrt(width))
        self.weight_imag = nn.Parameter(torch.randn(width, width, modes+1)/math.sqrt(width))

    def forward(self, u):
        spectrum = torch.fft.rfft(u, dim=-1, norm='ortho')
        count = min(spectrum.shape[-1], self.modes+1)
        w = torch.complex(self.weight_real, self.weight_imag)[..., :count]
        result = torch.zeros_like(spectrum)
        result[..., :count] = torch.einsum('bik,iok->bok', spectrum[..., :count], w)
        return torch.fft.irfft(result, n=u.shape[-1], dim=-1, norm='ortho')


class SpatialEncoder(nn.Module):
    def __init__(self, channels, width, depth, modes, ndim):
        super().__init__()
        conv = nn.Conv1d if ndim == 1 else nn.Conv2d
        spectral = Spectral1D if ndim == 1 else SymmetricSpectralMixer
        self.lift = conv(channels, width, 1)
        self.local = nn.ModuleList([conv(width, width, 3, padding=1, padding_mode='circular') for _ in range(depth)])
        self.spectral = nn.ModuleList([spectral(width, modes) for _ in range(depth)])

    def forward(self, features):
        z = F.gelu(self.lift(features))
        for local, spectral in zip(self.local, self.spectral):
            z = F.gelu(local(z)+spectral(z))
        return z


class LocalEncoder(nn.Module):
    """Horizon-independent pixelwise MLP with a spatial channel convention."""
    def __init__(self, channels, width, depth, ndim):
        super().__init__()
        conv = nn.Conv1d if ndim == 1 else nn.Conv2d
        layers = []
        for _ in range(depth):
            layers.extend((conv(channels, width, 1), nn.SiLU()))
            channels = width
        self.body = nn.Sequential(*layers)

    def forward(self, features):
        return self.body(features)


def zero_head(head):
    nn.init.zeros_(head.weight)
    if head.bias is not None:
        nn.init.zeros_(head.bias)


@dataclass(frozen=True)
class Encoding:
    owner: int
    u: torch.Tensor
    input_ref: torch.Tensor
    input_version: int | None
    eq: object
    geom: object
    target: str
    versions: tuple
    physics_versions: tuple
    features: torch.Tensor
    stats: dict
    latent: torch.Tensor | None = None
    temporal: object = None
    source_fields: torch.Tensor | None = None
    source_coefficients: torch.Tensor | None = None
    pooled: torch.Tensor | None = None
    inference_snapshots: tuple = ()
    mean_coefficients: torch.Tensor | None = None
    source_spectra: torch.Tensor | None = None
    diffusion_symbol: torch.Tensor | None = None
    remainder: torch.Tensor | None = None


def _physics_versions(eq):
    return tuple((id(x), _version(x)) for x in (eq.kappa, eq.reaction_rate) if torch.is_tensor(x))


def _version(value):
    try:
        return value._version
    except RuntimeError:
        # Inference tensors deliberately omit PyTorch's version counter. Their
        # cache contract is still checked using an explicitly retained value
        # snapshot, and that additional comparison remains in measured calls.
        return None


class ResponseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.requested_family = str(config.get('family', 'coupled_source'))
        self.family = ALIASES.get(self.requested_family, self.requested_family)
        if self.family not in ALL_FAMILIES:
            raise ValueError(f'Unsupported Response family {self.requested_family}')
        self.ndim = int(config.get('ndim', 2))
        self.width = int(config.get('width', 16)); self.depth = int(config.get('depth', 2))
        self.modes = int(config.get('modes', 4)); self.rank = int(config.get('rank', 4))
        self.mean_modes = int(config.get('mean_modes', 4))
        self.t_ref = float(config.get('t_ref', .1))
        self.anchor_name = config.get('anchor', 'df')
        self.feature_mode = config.get('feature_mode', 'phase')
        self.mean_parameterization = config.get('mean_parameterization', 'unrestricted')
        self.source_time_basis = config.get('source_time_basis', 'rational')
        self.source_feedback = config.get('source_feedback', 'full')
        self.source_decoder = config.get('source_decoder', 'quadrature')
        self.source_representation = config.get('source_representation', 'learned_basis')
        self.learned_time_rates = bool(config.get('learned_time_rates', False))
        self.numerical_remainder = bool(config.get('numerical_remainder', False))
        self.assembly = config.get('assembly', 'ordinary')
        self.auxiliary_source_head = bool(config.get('auxiliary_source_head', False))
        self.source_order = int(config['source_order']) if 'source_order' in config else None
        self.gain_normalization = config.get('gain_normalization', 'legacy')
        self.temporal_basis = config.get('temporal_basis', 'mixed')
        self.time_order = config.get('time_order', 'target' if self.family in DIAGNOSTIC_FAMILIES else 'legacy')
        self.nodes = int(config.get('quadrature_nodes', 4)); self.mean_nodes = int(config.get('mean_nodes', 4))
        self.reaction_substeps = int(config.get('reaction_substeps', 8))
        if self.ndim not in (1, 2) or min(self.width, self.depth, self.modes) < 1 or self.rank not in (2, 4, 8):
            raise ValueError('Expected ndim 1/2, positive sizes, source rank 2/4/8')
        if self.family in DIAGNOSTIC_FAMILIES and self.rank not in (4, 8):
            raise ValueError('Adaptive endpoint dictionaries require rank 4 or 8')
        if self.gain_normalization not in ('legacy', 'frozen'):
            raise ValueError('Gain normalization must be legacy or frozen')
        if self.temporal_basis not in ('mixed', 'response'):
            raise ValueError('Temporal basis must be mixed or response')
        target_time_order('fd_same_grid', self.time_order)
        if self.mean_modes < 1 or self.mean_modes > 8:
            raise ValueError('Mean trajectory requires one to eight temporal coefficients')
        if not math.isfinite(self.t_ref) or self.t_ref <= 0:
            raise ValueError('t_ref must be finite and positive')
        if self.anchor_name not in ('df', 'shared_gl4', 'gl4'):
            raise ValueError('Unknown physical anchor')
        if self.mean_parameterization not in ('unrestricted', 'clock') or self.source_time_basis not in ('rational', 'exponential', 'polynomial'):
            raise ValueError('Invalid mean or source time parameterization')
        if self.source_feedback not in ('full', 'frozen_background') or 'source_mode' in config:
            raise ValueError('Use source_feedback full/frozen_background; source_mode is not an implemented arm')
        if self.auxiliary_source_head and self.family not in ('direct_fno', 'residual_fno'):
            raise ValueError('An auxiliary source readout is only defined for direct/residual FNO')
        if self.source_order not in (None, 0, 1, 2):
            raise ValueError('Source onset order must be 0, 1 or 2')
        if self.source_decoder not in ('quadrature', 'cached_spectral') or self.source_representation not in ('learned_basis', 'fixed_physical', 'direct_interpolation'):
            raise ValueError('Unknown source decoder or representation')
        if self.source_representation == 'fixed_physical' and self.feature_mode == 'power':
            raise ValueError('A fixed phase-sensitive physical source basis is not a power-only ablation')
        if self.assembly not in ('ordinary', 'fp64'):
            raise ValueError('Assembly must be ordinary or fp64')
        if self.learned_time_rates and (self.source_time_basis == 'polynomial' or self.source_representation == 'direct_interpolation'):
            raise ValueError('Learned rates require rational/exponential basis channels')
        if self.family != 'coupled_source' and (self.source_decoder != 'quadrature' or self.source_representation != 'learned_basis' or self.learned_time_rates):
            raise ValueError('Source decoder controls require coupled_source')
        if self.family == 'linear_response' and (self.numerical_remainder or self.assembly != 'ordinary' or self.gain_normalization != 'legacy'):
            raise ValueError('Learned assembly/remainder/gain controls are not defined for linear_response')
        quadrature(self.nodes); quadrature(self.mean_nodes)
        self.normalization = FrozenNormalization(config.get('normalization'))
        # No additional persistent buffers in historical defaults. Explicit new
        # variants bind their finite, train-frozen gains to checkpoint bytes.
        if self.gain_normalization == 'frozen':
            normalization = config.get('normalization', {})
            for key in ('mean_gain', 'centered_gain'):
                value = float(normalization.get(key, float('nan')))
                fp32 = torch.tensor(value, dtype=torch.float32)
                if not math.isfinite(value) or value <= 0 or not bool(torch.isfinite(fp32)) or not bool(fp32 > 0):
                    raise ValueError('Frozen gain normalization requires positive finite '+key)
                self.register_buffer(key, torch.tensor(value, dtype=torch.float64))
        conv = nn.Conv1d if self.ndim == 1 else nn.Conv2d
        if self.family in ('direct_fno', 'residual_fno'):
            self.encoder = SpatialEncoder(FEATURE_CHANNELS+1, self.width, self.depth, self.modes, self.ndim)
            # A spatial bias vanishes under P0 and would be an unidentifiable
            # parameter. The separately learned mean owns the DC component.
            self.spatial_head = conv(self.width, 1, 1, bias=False)
            self.mean_head = nn.Linear(self.width, 1)
            zero_head(self.spatial_head); zero_head(self.mean_head)
            if self.auxiliary_source_head:
                self.auxiliary_head = conv(self.width, 1, 1, bias=False)
                zero_head(self.auxiliary_head)
        elif self.family == 'temporal':
            self.temporal_core = TemporalMLP(FEATURE_CHANNELS, modes=self.rank, width=self.width,
                hidden_layers=self.depth, t_ref=self.t_ref, U_ref=1.)
        elif self.family in DIAGNOSTIC_FAMILIES:
            encoder = LocalEncoder if self.family == 'adaptive_temporal' else SpatialEncoder
            self.encoder = (encoder(FEATURE_CHANNELS, self.width, self.depth, self.ndim)
                            if self.family == 'adaptive_temporal' else
                            encoder(FEATURE_CHANNELS, self.width, self.depth, self.modes, self.ndim))
            self.spatial_head = conv(self.width, self.rank, 1, bias=False)
            self.mean_head = nn.Linear(self.width, self.rank)
            zero_head(self.spatial_head); zero_head(self.mean_head)
        elif self.family == 'coupled_source':
            self.encoder = SpatialEncoder(FEATURE_CHANNELS, self.width, self.depth, self.modes, self.ndim)
            # Nonzero random spatial features; only final temporal amplitudes
            # and mean readout start at zero. This preserves head gradients.
            if self.source_representation == 'direct_interpolation':
                self.interpolation_head = conv(self.width, self.rank, 1)
                zero_head(self.interpolation_head)
            else:
                if self.source_representation == 'learned_basis':
                    self.source_fields = conv(self.width, self.rank, 1)
                self.source_head = nn.Linear(self.width, self.rank)
                zero_head(self.source_head)
            if self.numerical_remainder:
                self.remainder_head = conv(self.width, 1, 1)
                zero_head(self.remainder_head)
            # A reusable, state-dependent mean trajectory, with capacity held
            # fixed when the centered source rank J changes. Four modes can
            # express h^3, h^4 and higher numerical-subflow remainder onsets.
            self.mean_head = nn.Linear(self.width, self.mean_modes)
            zero_head(self.mean_head)
            self.register_buffer('source_rates', torch.logspace(-1, 1, self.rank))
            if self.learned_time_rates:
                self.source_rate_logits = nn.Parameter(torch.logit((self.source_rates-.01)/19.99))
        if self.numerical_remainder and self.family != 'coupled_source':
            self.remainder_head = (nn.Linear(self.width, 1) if self.family == 'temporal' else conv(self.width, 1, 1))
            zero_head(self.remainder_head)
        if config.get('frozen_encoder', False):
            encoder = getattr(self, 'encoder', None)
            if encoder is None:
                raise ValueError('Frozen encoder control requires a spatial encoder')
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)

    def _versions(self):
        return tuple((id(x), _version(x), x.device, x.dtype) for x in list(self.parameters())+list(self.buffers()))

    def architecture_metadata(self):
        return dict(family=self.family, requested_family=self.requested_family,
            alias_of=self.family if self.requested_family in ALIASES else None,
            ownership='Analytic control' if self.family == 'linear_response' else 'Theirs' if self.family in ('direct_fno', 'residual_fno') else 'Ours',
            anchor=None if self.family == 'direct_fno' else self.anchor_name,
            initial_parent='identity' if self.family == 'direct_fno' else
                'initial-variance conditional clock plus centered anchor' if self.family == 'coupled_source' and self.mean_parameterization == 'clock' else self.anchor_name,
            finite_time_model=self.family == 'coupled_source', source_channels=self.rank,
            mean_temporal_modes=self.mean_modes if self.family == 'coupled_source' else None,
            source_onset_order=self.source_order if self.source_order is not None else '2 same-grid; 1 resolved-initial projected-fine; hidden closure must declare 0', source_time_basis=self.source_time_basis,
            mean_parameterization=self.mean_parameterization, feature_mode=self.feature_mode,
            feature_channels=FEATURE_CHANNELS,feature_names=list(FEATURE_NAMES),
            physical_band_edges_cycles_per_length=list(PHYSICAL_BAND_EDGES),
            power_feature_substitutions='Extrema replaced by mean +/- RMS, spatial bands by band RMS; no input phase' if self.feature_mode == 'power' else None,
            source_feedback=self.source_feedback, source_decoder=self.source_decoder,
            source_representation=self.source_representation, learned_time_rates=self.learned_time_rates,
            source_rate_bounds=[.01,20.] if self.learned_time_rates else None,
            interpolation_contract='C1 piecewise cubic smoothstep in s/(t_ref+s), fixed equally spaced knots, bounded convex snapshot interpolation; second derivative jumps at knots' if self.source_representation == 'direct_interpolation' else None,
            source_transform_contract='encode J source FFTs once; decode one inverse FFT for centered response; anchor and mean-path FFTs still charged' if self.source_decoder == 'cached_spectral' else 'one source heat transform pair per quadrature node',
            numerical_remainder=self.numerical_remainder,
            numerical_remainder_scope='Separate h/(t_ref+h) channel without ideal-physics masks; permits arithmetic/subflow defects, not a new physical interaction' if self.numerical_remainder else None,
            assembly=self.assembly, restart_precision='Model parameter dtype on every re-encoding; final FP64 output retained between reported endpoints' if self.assembly == 'fp64' else 'Model arithmetic dtype', output_precision='float64 final addition, same deployed anchor and learned correction' if self.assembly == 'fp64' else 'model arithmetic dtype',
            propagation='learned mean trajectory' if self.source_feedback == 'full' else 'frozen homogeneous logistic background; coupled-source ablation, not exact linear dynamics',
            phase_blind_limitation='Centered learned correction is identically zero from constant power-only features' if self.feature_mode == 'power' else None,
            original_temporal_core='tdn.models.temporal.TemporalMLP; adapted physical inputs and frozen channel scales' if self.family == 'temporal' else None,
            reusable_encoding=self.family in ('temporal', 'coupled_source', *DIAGNOSTIC_FAMILIES),
            gain_normalization=self.gain_normalization,
            frozen_gains={key: float(getattr(self, key).detach().cpu()) for key in ('mean_gain', 'centered_gain')} if self.gain_normalization == 'frozen' else None,
            endpoint_temporal_basis=self.temporal_basis if self.family in DIAGNOSTIC_FAMILIES else None,
            endpoint_time_order=self.time_order if self.family in DIAGNOSTIC_FAMILIES else 'legacy cubic' if self.family == 'temporal' else None,
            endpoint_time_dictionary_rank=self.rank if self.family in DIAGNOSTIC_FAMILIES else None,
            gain_scope='Positive train-frozen scalar basis rescaling; no pointwise division by gate, time or labels',
            query_contract='scalar, per-parent [B], or per-parent query table [B,Q]; never silent spatial broadcasting',
            cache_validation='Tensor version checks; explicit value comparison for unversioned inference tensors',
            null_contract='h=0,r=0; constant and kappa=0 correction nulls only FD exact-subflow splitting; Galerkin numerical reaction remainder retained',
            learned_clock_scope='bounded nodal fd_same_grid only; not a Galerkin positivity guarantee',
            normalization=self.normalization.metadata(), quadrature_nodes=self.nodes, mean_nodes=self.mean_nodes,
            quadrature_claim='Finite quadrature approximation, convergence must be measured',
            inference_teacher_inputs=False, output_clipping=False,
            auxiliary_source_head=self.auxiliary_source_head,
            auxiliary_source_scope='Training-only centered full-source supervision, decoder unchanged; extra stored parameters and training work charged' if self.auxiliary_source_head else None,
            stored_parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))

    def encode(self, u, eq, geom, target):
        if geom.ndim != self.ndim:
            raise ValueError('Geometry dimension differs from the model architecture')
        if self.mean_parameterization == 'clock' and self.family == 'coupled_source':
            if target != 'fd_same_grid' or bool(((u < 0) | (u > 1)).any()):
                raise ValueError('Clock requires bounded nodal fd_same_grid input; no clipping is performed')
        # Snapshot makes later modifications unable to corrupt an encoded
        # trajectory. The original tensor's version is still checked below.
        state = (u.to(next(self.parameters()).dtype) if self.assembly == 'fp64' else u).clone()
        features, stats = build_features(state, eq, geom, target, self.normalization,
                                         t_ref=self.t_ref, mode=self.feature_mode)
        latent = self.encoder(features) if self.family in ('coupled_source', *DIAGNOSTIC_FAMILIES) else None
        temporal = self.temporal_core.encode(features.movedim(1, -1)) if self.family == 'temporal' else None
        fields = coeff = pooled = mean_coefficients = spectra = symbol = remainder = None
        if self.family in DIAGNOSTIC_FAMILIES:
            pooled = latent.mean(tuple(range(2, latent.ndim)))
            temporal = dict(spatial=n.center(self.spatial_head(latent)),
                            mean=self.mean_head(pooled).reshape(u.shape[0], self.rank, *([1]*self.ndim)))
            if self.numerical_remainder:
                remainder = self.remainder_head(latent)
        elif self.family == 'temporal' and self.numerical_remainder:
            # The historical core remains unchanged. This explicit diagnostic
            # arm charges its extra local feature pass for the separate readout.
            local = self.temporal_core.body(features.movedim(1, -1))
            remainder = self.remainder_head(local).movedim(-1, 1)
        if self.family == 'coupled_source':
            pooled = latent.mean(tuple(range(2, latent.ndim)))
            if self.source_representation == 'direct_interpolation':
                fields = self.interpolation_head(latent)
                coeff = torch.ones((u.shape[0], self.rank, *([1]*self.ndim)), dtype=state.dtype, device=state.device)
            else:
                fields = self._fixed_fields(state, eq, geom, target) if self.source_representation == 'fixed_physical' else self.source_fields(latent)
                coeff = self.source_head(pooled).reshape(u.shape[0], self.rank, *([1]*self.ndim))
            fields = n.center(fields)
            if self.source_decoder == 'cached_spectral':
                spectra = torch.fft.fftn(fields, dim=n.axes(fields), norm='forward')
                dc_mask = torch.ones_like(fields)
                dc_mask[(slice(None), slice(None), *([0]*self.ndim))] = 0
                spectra = spectra*dc_mask
                symbol = n.diffusion_symbol(state, eq, geom, target)
            if self.numerical_remainder:
                remainder = self.remainder_head(latent)
            mean_coefficients = self.mean_head(pooled)
        checked = [u, *self.parameters(), *self.buffers()]+[x for x in (eq.kappa,eq.reaction_rate) if torch.is_tensor(x)]
        snapshots = tuple((x, x.clone()) for x in checked if _version(x) is None)
        return Encoding(id(self), state, u, _version(u), eq, geom, target, self._versions(),
                        _physics_versions(eq), features, stats, latent, temporal, fields, coeff, pooled, snapshots, mean_coefficients, spectra, symbol, remainder)

    def _check(self, encoded):
        if not isinstance(encoded, Encoding) or encoded.owner != id(self):
            raise ValueError('Encoding belongs to a different model')
        if encoded.versions != self._versions() or encoded.input_version != _version(encoded.input_ref) or encoded.physics_versions != _physics_versions(encoded.eq) or any(not torch.equal(x, snapshot) for x, snapshot in encoded.inference_snapshots):
            raise ValueError('Stale encoding: state, physics, model parameters, buffers or device changed; re-encode')

    def _anchor(self, encoded, h):
        return n.anchor(encoded.u, h, encoded.eq, encoded.geom, encoded.target,
                        name=self.anchor_name, reaction_substeps=self.reaction_substeps)

    def _gain(self, channel, like):
        return getattr(self, channel+'_gain').to(like) if self.gain_normalization == 'frozen' else 1.

    def _output_scales(self, like):
        return (self.normalization.mean_scale/self._gain('mean', like),
                self.normalization.centered_scale/self._gain('centered', like))

    def _envelope(self, t, encoded):
        tau = n.scalar(t, encoded.u)/self.t_ref
        order = self.source_order if self.source_order is not None else (1 if encoded.target == 'galerkin_projected_fine' else 2)
        return torch.ones_like(tau) if order == 0 else tau**order/(1+tau**order)

    def source_terms(self, encoded, s):
        """Learned approximation of the FULL centered defect forcing at time s."""
        self._check(encoded)
        if self.family in ('direct_fno', 'residual_fno') and self.auxiliary_source_head:
            step = n.validate(encoded.u, s, encoded.geom, encoded.target)
            tau = (step/self.t_ref).expand_as(encoded.u)
            latent = self.encoder(torch.cat((encoded.features, tau/(1+tau)), dim=1))
            value = n.center(self.auxiliary_head(latent))
            return value*self.normalization.centered_scale/self.t_ref*self._envelope(step, encoded)*correction_gate(encoded.stats, encoded.target)
        if self.family != 'coupled_source':
            raise ValueError('Source channels belong to the coupled model')
        value = (encoded.source_fields*self._source_weights(encoded, s)).sum(dim=1, keepdim=True)
        return n.center(value)

    def _fixed_fields(self, u, eq, geom, target):
        """Predeclared physical channels, not teacher-selected or learned fields."""
        v = n.center(u)/self.normalization.full_scale
        q = n.center(n.product(v,v,geom,target))
        fields = [v,q]
        if self.rank >= 4:
            hot=n.heat(v,.25*self.t_ref,eq,geom,target)
            fields.extend([hot,n.center(n.product(v,q,geom,target))])
        if self.rank == 8:
            fields.extend([n.heat(v,self.t_ref,eq,geom,target),n.heat(v,2*self.t_ref,eq,geom,target),
                           n.heat(q,.25*self.t_ref,eq,geom,target),n.center(n.product(v,hot,geom,target))])
        return torch.cat(fields,dim=1)

    def _time_basis(self, encoded, s):
        tau = n.validate(encoded.u, s, encoded.geom, encoded.target)/self.t_ref
        shape = (1,self.rank,*([1]*self.ndim))
        if self.source_representation == 'direct_interpolation':
            # Continuous first derivative at every knot, no hidden clamp of
            # time or source. Nonnegative horizons map into [0,1).
            x = tau/(1+tau)*(self.rank-1)
            left = torch.floor(x).long().clamp(max=self.rank-2)
            theta = x-left
            w = theta.square()*(3-2*theta)
            indices = torch.arange(self.rank,device=tau.device).reshape(shape)
            basis = (indices == left)*(1-w)+(indices == left+1)*w
        else:
            rates = (.01+19.99*torch.sigmoid(self.source_rate_logits)) if self.learned_time_rates else self.source_rates
            rates = rates.reshape(shape)
            if self.source_time_basis == 'rational':
                basis = 1/(1+rates*tau)
            elif self.source_time_basis == 'exponential':
                basis = torch.exp(-rates*tau)
            else:
                basis = (1+tau)**torch.arange(self.rank, device=tau.device, dtype=tau.dtype).reshape(shape)
        return basis

    def _source_weights(self, encoded, s):
        return encoded.source_coefficients*self._time_basis(encoded,s)*self.normalization.centered_scale/self._gain('centered', encoded.u)/self.t_ref*self._envelope(s,encoded)*correction_gate(encoded.stats,encoded.target)

    def source_components(self, encoded, s):
        self._check(encoded)
        if self.family != 'coupled_source':
            raise ValueError('Source components require coupled_source')
        return dict(fields=encoded.source_fields, coefficients=encoded.source_coefficients,
                    temporal_basis=self._time_basis(encoded,s), effective_coefficients=self._source_weights(encoded,s),
                    representation=self.source_representation, decoder=self.source_decoder)

    def _spectral_response(self, encoded, h, mean_path):
        """Same GL formula as source_response, factoring all spatial transport.

        Full FFT coefficients preserve real/Hermitian/Nyquist conventions.
        No Parseval weights are needed for synthesis (they apply to norms).
        Mean-path quadrature is unchanged and explicitly remains real work.
        """
        kernel = torch.zeros_like(encoded.source_spectra.real)
        for x,w in zip(*quadrature(self.nodes)):
            s=h*x
            propagator=mean_propagator(mean_path,s,h,encoded.eq,encoded.u,nodes=self.mean_nodes)
            kernel = kernel+w*propagator*torch.exp((h-s)*encoded.diffusion_symbol)*self._source_weights(encoded,s)
        spectrum=(encoded.source_spectra*kernel).sum(dim=1,keepdim=True)*h
        return n.center(torch.fft.ifftn(spectrum,dim=n.axes(encoded.u),norm='forward').real)

    def _assemble(self, base, *corrections):
        if self.assembly == 'fp64':
            value=base.double()
            for correction in corrections:
                value=value+correction.double()
            return value
        value=base
        for correction in corrections:
            value=value+correction
        return value

    def _unrestricted_mean_correction(self, encoded, t):
        tau=n.scalar(t,encoded.u)/self.t_ref
        time=tau.expand(encoded.u.shape[0],1,*([1]*self.ndim)).reshape(encoded.u.shape[0],1)
        powers=(time/(1+time))**torch.arange(self.mean_modes,dtype=time.dtype,device=time.device)[None]
        raw=(encoded.mean_coefficients*powers).sum(1).reshape(encoded.u.shape[0],1,*([1]*self.ndim))
        return self.normalization.mean_scale/self._gain('mean', encoded.u)*raw*tau*self._envelope(t,encoded)*correction_gate(encoded.stats,encoded.target)

    def _endpoint_parts(self, encoded, t):
        """Assemble endpoint terms once; retain historical arithmetic ordering."""
        mean_scale, centered_scale = self._output_scales(encoded.u)
        remainder_raw = encoded.remainder
        if self.family in ('direct_fno', 'residual_fno'):
            time = (t/self.t_ref).expand_as(encoded.u)
            latent = self.encoder(torch.cat((encoded.features, time/(1+time)), dim=1))
            raw_centered = n.center(self.spatial_head(latent))
            raw_mean = self.mean_head(latent.mean(tuple(range(2, latent.ndim)))).reshape(encoded.u.shape[0], 1, *([1]*self.ndim))
            if self.numerical_remainder:
                remainder_raw = self.remainder_head(latent)
            if self.family == 'direct_fno':
                base = encoded.u
                factor = t/self.t_ref
                mean_scale = self.normalization.full_scale/self._gain('mean', encoded.u)
                centered_scale = self.normalization.full_scale/self._gain('centered', encoded.u)
                if self.gain_normalization == 'legacy':
                    physical = factor*self.normalization.full_scale*(raw_centered+raw_mean)
                else:
                    physical = factor*(centered_scale*raw_centered+mean_scale*raw_mean)
            else:
                base = self._anchor(encoded, t)
                factor = (t/self.t_ref)*self._envelope(t, encoded)*correction_gate(encoded.stats, encoded.target)
                physical = factor*(centered_scale*raw_centered+mean_scale*raw_mean)
            mean_correction, centered_correction = factor*mean_scale*raw_mean, factor*centered_scale*raw_centered
        elif self.family == 'temporal':
            step = t.reshape(encoded.u.shape[0] if t.numel() > 1 else 1, *([1]*(self.ndim+2)))
            raw = self.temporal_core.decode(encoded.temporal, step).movedim(-1, 1)
            raw_mean, raw_centered = n.mean(raw), n.center(raw)
            factor = correction_gate(encoded.stats, encoded.target)
            physical = (mean_scale*raw_mean+centered_scale*raw_centered)*factor
            mean_correction, centered_correction = factor*mean_scale*raw_mean, factor*centered_scale*raw_centered
            base = self._anchor(encoded, t)
        elif self.family in DIAGNOSTIC_FAMILIES:
            tau = (t/self.t_ref).expand(encoded.u.shape[0], 1, *([1]*self.ndim))
            basis = endpoint_basis(tau, order=target_time_order(encoded.target, self.time_order), rank=self.rank, kind=self.temporal_basis)
            raw_mean, raw_centered = encoded.temporal['mean'], encoded.temporal['spatial']
            factor = correction_gate(encoded.stats, encoded.target)
            mean_correction = factor*mean_scale*(raw_mean*basis).sum(1, keepdim=True)
            centered_correction = factor*centered_scale*n.center((raw_centered*basis).sum(1, keepdim=True))
            physical = mean_correction+centered_correction
            base = self._anchor(encoded, t)
        else:
            raise ValueError('Endpoint terms are unavailable for this family')
        # Arithmetic/subflow discrepancies do not inherit ideal splitting
        # nulls. The separate branch still vanishes smoothly at h=0.
        remainder = (torch.zeros_like(base) if remainder_raw is None else
                     (t/(self.t_ref+t))*(self.normalization.mean_scale*n.mean(remainder_raw)+self.normalization.centered_scale*n.center(remainder_raw)))
        return dict(base=base, physical=physical, mean_correction=mean_correction,
                    centered_correction=centered_correction, numerical_remainder=remainder,
                    raw_mean=raw_mean, raw_centered=raw_centered)

    def mean_path(self, encoded, s):
        self._check(encoded)
        if self.family != 'coupled_source':
            raise ValueError('Learned mean trajectory belongs to the coupled model')
        t = n.validate(encoded.u, s, encoded.geom, encoded.target)
        time = (t/self.t_ref).expand(encoded.u.shape[0], 1, *([1]*self.ndim)).reshape(encoded.u.shape[0], 1)
        def raw_at(tau):
            coordinate = tau/(1+tau)
            powers = coordinate**torch.arange(self.mean_modes, dtype=tau.dtype, device=tau.device)[None]
            return (encoded.mean_coefficients*powers).sum(dim=1).reshape(encoded.u.shape[0], 1, *([1]*self.ndim))
        raw = raw_at(time)
        if self.mean_parameterization == 'unrestricted':
            return n.mean(self._anchor(encoded,t))+self._unrestricted_mean_correction(encoded,t)
        m0, variance = encoded.stats['mean'], encoded.stats['variance']
        interior = (m0 > 0) & (m0 < 1)
        safe_m = torch.where(interior, m0, torch.full_like(m0, .5))
        z0 = variance/(safe_m*(1-safe_m))
        # Roundoff tolerance accepts an endpoint variance; this does not clamp
        # the physical state. The head uses a convex reweighting of z0 and a
        # bounded learned clock rate, preserving initial z0 at raw=0.
        if bool((z0 > 1+32*torch.finfo(encoded.u.dtype).eps).any()):
            raise ValueError('Initial variance violates the bounded nodal clock contract')
        z0 = z0.clamp(0, 1)
        clock_integral = torch.zeros_like(m0)
        for x, w in zip(*quadrature(self.mean_nodes)):
            tx = time*x
            adjustment = raw_at(tx)
            # Logistic odds adjustment with exact z0=0 and z0=1 branches.
            inside_z = (z0 > 0) & (z0 < 1)
            safe_z = torch.where(inside_z, z0, torch.full_like(z0, .5))
            z = torch.where(inside_z, torch.sigmoid(torch.logit(safe_z)+adjustment), z0)
            clock_integral = clock_integral+w*(1-z)
        result = torch.sigmoid(torch.logit(safe_m)+n.scalar(encoded.eq.reaction_rate, encoded.u)*t*clock_integral)
        result = torch.where(interior, result, m0)
        # FD reaction-only and linear-heat parents are exact physical nulls;
        # the constant-z clock is not an exact reaction-only mean trajectory.
        gate = correction_gate(encoded.stats, encoded.target)
        return torch.where(interior, torch.where(gate == 0, n.mean(self._anchor(encoded, t)), result), m0)

    def _single(self, encoded, h):
        t = n.validate(encoded.u, h, encoded.geom, encoded.target)
        if self.family == 'linear_response':
            return self._anchor(encoded, t)+homogeneous_residual_response(encoded.u, t,
                encoded.eq, encoded.geom, encoded.target, name=self.anchor_name,
                nodes=self.nodes, reaction_substeps=self.reaction_substeps,
                create_graph=torch.is_grad_enabled())
        if self.family in ('direct_fno', 'residual_fno', 'temporal', *DIAGNOSTIC_FAMILIES):
            parts = self._endpoint_parts(encoded, t)
            value = self._assemble(parts['base'], parts['physical'])
            if self.numerical_remainder:
                value = self._assemble(value, parts['numerical_remainder'])
            if self.family in DIAGNOSTIC_FAMILIES or self.gain_normalization != 'legacy' or self.assembly != 'ordinary' or self.numerical_remainder:
                value = torch.where(t == 0, encoded.input_ref.to(value.dtype), value)
            return value
        parts = self._coupled_parts(encoded, t)
        value = self._assemble(parts['base'], parts['mean_correction'], parts['centered_correction'], parts['numerical_remainder'])
        return torch.where(t == 0, encoded.input_ref.to(value.dtype), value)

    def _coupled_parts(self, encoded, t):
        mean_for_propagator = (lambda s: self.mean_path(encoded, s)) if self.source_feedback == 'full' else (
            lambda s: background(encoded.stats['mean'], s, encoded.eq.reaction_rate)['c'])
        if self.source_decoder == 'cached_spectral':
            d = self._spectral_response(encoded,t,mean_for_propagator)
        else:
            d = source_response(lambda s: self.source_terms(encoded, s), t, encoded.eq,
                encoded.geom, encoded.target, mean_for_propagator,
                nodes=self.nodes, mean_nodes=self.mean_nodes)
        base = self._anchor(encoded, t)
        # Algebraically m_theta + P0(a) + d, but this assembly retains the
        # literal deployed anchor when both learned heads are zero. Subtracting
        # and re-adding its mean first could introduce a spurious FP32 ULP.
        remainder = torch.zeros_like(base) if encoded.remainder is None else (t/(self.t_ref+t))*(self.normalization.mean_scale*n.mean(encoded.remainder)+self.normalization.centered_scale*n.center(encoded.remainder))
        mean_delta = self._unrestricted_mean_correction(encoded,t) if self.mean_parameterization == 'unrestricted' else self.mean_path(encoded,t)-n.mean(base)
        return dict(base=base, mean_correction=mean_delta, centered_correction=d,
                    numerical_remainder=remainder, raw_mean=encoded.mean_coefficients,
                    raw_centered=encoded.source_fields*encoded.source_coefficients)

    def correction_diagnostics(self, u, h, eq, geom, target):
        """Detached pre-addition terms; post-hoc work, never a training target.

        One scalar/per-parent query is supported. Dictionary raw channels are
        reported before time/gate scaling; historical temporal raw output
        already contains its analytic time response. Assembly impact compares
        final addition only, retaining the same network and deployed anchor.
        """
        if self.family == 'linear_response':
            raise ValueError('Learned correction diagnostics exclude linear_response')
        with torch.no_grad():
            encoded = self.encode(u, eq, geom, target)
            if torch.as_tensor(h).ndim > 1:
                raise ValueError('Correction diagnostics require scalar or per-parent horizons')
            t = n.validate(encoded.u, h, geom, target)
            if self.family == 'coupled_source':
                parts = self._coupled_parts(encoded, t)
                corrections = (parts['mean_correction'], parts['centered_correction'], parts['numerical_remainder'])
            else:
                parts = self._endpoint_parts(encoded, t)
                corrections = ((parts['physical'], parts['numerical_remainder']) if self.numerical_remainder else (parts['physical'],))
            ordinary = parts['base']
            high = ordinary.double()
            for correction in corrections:
                ordinary = ordinary+correction
                high = high+correction.double()
            output = high if self.assembly == 'fp64' else ordinary
            if self.family in (*DIAGNOSTIC_FAMILIES, 'coupled_source') or self.gain_normalization != 'legacy' or self.assembly != 'ordinary' or self.numerical_remainder:
                output = torch.where(t == 0, u.to(output.dtype), output)
            gains = correction_gains(encoded.u, t, eq, geom, target, family=self.family,
                                     config={**self.config, 'normalization': self.normalization.metadata()})
            total = parts['mean_correction']+parts['centered_correction']+parts['numerical_remainder']
            return {key: value.detach() for key, value in dict(
                raw_mean=parts['raw_mean'], raw_centered=parts['raw_centered'],
                mean_correction=parts['mean_correction'], centered_correction=parts['centered_correction'],
                numerical_remainder=parts['numerical_remainder'], total_correction=total,
                mean_gain=gains['mean_gain']/self._gain('mean', encoded.u),
                centered_gain=gains['centered_gain']/self._gain('centered', encoded.u),
                assembly_difference=high-ordinary.double(), output=output).items()}

    def decode(self, encoded, h):
        self._check(encoded)
        horizons = torch.as_tensor(h, dtype=encoded.u.dtype, device=encoded.u.device)
        if horizons.ndim == 2:
            if horizons.shape[0] != encoded.u.shape[0] or horizons.shape[1] < 1:
                raise ValueError('Query horizons must have shape [B,Q] with Q positive')
            return torch.stack([self._single(encoded, horizons[:, i]) for i in range(horizons.shape[1])], dim=1)
        if horizons.ndim > 1 or (horizons.ndim == 1 and horizons.numel() != encoded.u.shape[0]):
            raise ValueError('Horizons must be scalar, [B], or [B,Q]')
        return self._single(encoded, horizons)

    def forward(self, u, h, eq, geom, target):
        return self.decode(self.encode(u, eq, geom, target), h)

    def affine_features(self, u, h, eq, geom, target):
        """Exact linear readout design with a frozen random spatial encoder.

        Basis shape is [2*width+1,B,1,*grid]. This claim is intentionally not
        applied to the coupled decoder, whose mean propagator is nonlinear in
        learned parameters. Feature extraction and anchor costs still apply.
        """
        if self.family not in ('direct_fno', 'residual_fno') or self.numerical_remainder or any(p.requires_grad for p in self.encoder.parameters()):
            raise ValueError('Affine features require a frozen direct/residual FNO encoder')
        encoded = self.encode(u, eq, geom, target)
        state = encoded.u
        t = n.validate(state, h, geom, target)
        time = (t/self.t_ref).expand_as(state)
        z = self.encoder(torch.cat((encoded.features, time/(1+time)), dim=1))
        pooled = z.mean(tuple(range(2, z.ndim)), keepdim=True)
        spatial = z-pooled
        if self.family == 'direct_fno':
            base = state
            spatial_scale = (t/self.t_ref)*self.normalization.full_scale/self._gain('centered', state)
            mean_scale = (t/self.t_ref)*self.normalization.full_scale/self._gain('mean', state)
        else:
            base = self._anchor(encoded, t)
            scale = (t/self.t_ref)*self._envelope(t, encoded)*correction_gate(encoded.stats, target)
            spatial_scale = scale*self.normalization.centered_scale/self._gain('centered', state)
            mean_scale = scale*self.normalization.mean_scale/self._gain('mean', state)
        bases = [(spatial[:, j:j+1]*spatial_scale) for j in range(self.width)]
        bases += [(pooled[:, j:j+1]*mean_scale).expand_as(state) for j in range(self.width)]
        bases += [mean_scale.expand_as(state)]
        if self.assembly == 'fp64':
            base = base.double()
        if self.gain_normalization != 'legacy' or self.assembly != 'ordinary':
            base = torch.where(t == 0, u.to(base.dtype), base)
        return (base, torch.stack(bases).double()) if self.assembly == 'fp64' else (base, torch.stack(bases))

    def set_affine_coefficients(self, vector):
        if self.family not in ('direct_fno', 'residual_fno') or self.numerical_remainder or any(p.requires_grad for p in self.encoder.parameters()):
            raise ValueError('Coefficient assignment requires a frozen direct/residual FNO encoder')
        value = torch.as_tensor(vector, dtype=self.spatial_head.weight.dtype, device=self.spatial_head.weight.device)
        if value.shape != (2*self.width+1,) or not bool(torch.isfinite(value).all()):
            raise ValueError('Expected a finite readout coefficient vector of length 2*width+1')
        with torch.no_grad():
            self.spatial_head.weight.copy_(value[:self.width].reshape_as(self.spatial_head.weight))
            self.mean_head.weight.copy_(value[self.width:2*self.width].reshape_as(self.mean_head.weight))
            self.mean_head.bias.copy_(value[-1:])

    def step(self, u, h, eq, geom, target):
        value = self(u, h, eq, geom, target)
        return value, dict(family=self.family, target=target,
            finite=bool(torch.isfinite(value).all()), mean=n.mean(value).detach(),
            mean_parameterization=self.mean_parameterization,
            quadrature_nodes=self.nodes if self.family in ('coupled_source', 'linear_response') else 0,
            source_decoder=self.source_decoder, assembly=self.assembly,
            source_evaluations=(self.nodes+1 if self.source_decoder == 'quadrature' else 0) if self.family == 'coupled_source' else 0,
            source_inverse_transforms=(1 if self.source_decoder == 'cached_spectral' else self.nodes) if self.family == 'coupled_source' else 0,
            mean_path_evaluations=(self.nodes*self.mean_nodes+1 if self.source_feedback == 'full' else 1) if self.family == 'coupled_source' else 0,
            numerical_correctness='unverified per-call; independent tests and convergence required')

    def rollout(self, u, schedule, eq, geom, target, *, return_states=False):
        if len(schedule) == 0:
            raise ValueError('Rollout requires at least one nonnegative step')
        states = [u]
        for h in schedule:
            # A trajectory encoding is valid only from its current initial
            # state. Refresh after every accepted step, even repeated h.
            states.append(self(states[-1], h, eq, geom, target))
        return torch.stack(states, dim=1) if return_states else states[-1]


def create_model(config):
    return ResponseModel(config)
