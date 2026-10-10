"""Independent, finite adjacent-study declarations, never a portfolio override.

Full is a bounded pilot plus a precision-qualified fresh evaluation. A failed
precision or optimization gate produces NA, never an automatic larger campaign.
"""
from __future__ import annotations
import hashlib
import json
import math

SCHEMA = 'tdn.adjacent/v1'
HISTORICAL_PROFILES = ('smoke', 'development', 'full')
RESOLUTION_PROFILES = ('resolution-smoke', 'resolution-full', 'resolution64', 'resolution128')
PROFILES = HISTORICAL_PROFILES + RESOLUTION_PROFILES
NATIVE_FULL_PROFILES = frozenset(('full', 'resolution-full', 'resolution64', 'resolution128'))
DIAGNOSTICS = ('D01', 'D05', 'D02', 'D04', 'D03', 'D06', 'D07', 'D08')
HYPOTHESES = {
    'D01': 'Scalar, band, interaction-channel and cubic representation floors',
    'D02': 'Feature collisions require materially different responses',
    'D03': 'Grid, step, duration and complete accuracy-qualified cost',
    'D04': 'Signed high-to-low interactions and phase information',
    'D05': 'Quadrature versus missing interaction order',
    'D06': 'Autonomous transient fidelity and stability',
    'D07': 'Accuracy-qualified temporal reuse and refresh economics',
    'D08': 'Competitor-favorable and assumption-breaking workloads',
    'R01': 'Controlled finite-resolution rough-surface benchmark',
    'C01': 'Origin-of-interaction channels versus fitted and learned controls',
    'E01': 'Select cubic work before computing the cubic term',
    'E02': 'Signed modulation as a cheaper cubic spatial direction',
}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def _unit(name, kind, prior, seconds, *, device='cpu', **extra):
    minutes = min(45, max(8, math.ceil(seconds / 60) + (8 if device == 'cuda' else 5)))
    return dict(id=name, kind=kind, path='adjacent', dependencies=list(prior),
                seconds=seconds, device=device, cpus=4, mem_gib=32,
                walltime=f'00:{minutes:02d}:00', **extra)

def build_protocol(profile='smoke'):
    if profile in RESOLUTION_PROFILES:
        from .resolution_protocol import build_resolution_protocol
        return build_resolution_protocol(profile)
    if profile not in PROFILES:
        raise ValueError('Adjacent profile must be smoke, development or full')
    full, smoke = profile == 'full', profile == 'smoke'
    pilot_seconds = 2400 if full else 320 if smoke else 960
    exploration_seconds = pilot_seconds // 4
    p = dict(schema=SCHEMA, version=1, profile=profile,
        purpose='Adjacent origin-of-interaction study; no promotion or modification of the active portfolio',
        original_objective='Approximate coupled finite-time splitting defects with reusable temporal structure at lower complete accuracy-qualified cost',
        main_campaign_unchanged=True, automatic_budget_expansion=False,
        tracks=['discrete', 'continuum'], timing_repeats=3 if smoke else 7,
        mechanisms={key: dict(name=value, question=value) for key, value in HYPOTHESES.items()}, combinations={},
        seed={'smoke':71000000,'development':72000000,'full':73000000}[profile],
        diagnostics=dict(grid=8 if smoke else 16, horizon=.12, uncertainty_floor=1e-12,
            alpha_levels=[.1,.5,.9,1.] if smoke else [i/10 for i in range(1,11)],
            max_reference_substeps=256, teacher_precision='float64',
            same_grid_and_continuum_targets_separate=True),
        roughness=dict(alpha_levels=[i/10 for i in range(1,11)], generators=['ridge','multiscale2d'],
            finite_sums_are_smooth=True, alpha_one='finite-bandwidth stress; H=0 infinite sum not convergent',
            no_clipping=True, paired_grid_refinement=True, channel_axis_is_not_space=True),
        pilot=dict(grid=32 if full else 8 if smoke else 16,
            tracks=['discrete','continuum'], train_fields=32 if full else 2 if smoke else 8,
            val_fields=16 if full else 2 if smoke else 4,
            evaluation_fields=64 if full else 2 if smoke else 4,
            updates=200 if full else 2 if smoke else 24,
            seeds=[731001], horizons=[.06] if smoke else [.04,.12],
            endpoint_steps=[1,2,4,8], alpha_cycle=[i/10 for i in range(1,11)] if full else [.3,.7,1.],
            max_seconds=1200 if full else pilot_seconds, trial_seconds=90 if full else 15 if smoke else 45,
            reference_tolerance=1e-7, reference_substeps=8, max_substeps=256,
            seed_offset={'smoke':71000000,'development':72000000,'full':73000000}[profile]),
        confirmation=dict(maximum_parents=64, desired_log_halfwidth=math.log(1.15),
            confidence_level=.95, primary_comparisons=6, multiple_comparisons='Bonferroni across three predeclared contrasts on each of two separate equation tracks',
            planning='validation independent-parent paired log-error variability; freeze N before generating fresh fields',
            insufficient_precision='NA, retain all observations; never silently expand budget',
            primary_effect_ratio=1.10, practical_speedup=1.20, coverage_regression=.05,
            independent_unit='parent field, with phases/grids/schedules/roughness variants/seeds paired',
            primary_claims=['channel_neural vs channel_global','channel_neural vs quad2_conditioned','channel_neural vs band_gain'],
            inspected_confirmation_becomes_development=True),
        exploration=dict(prototype_ids=['E01','E02'], novel_in_literature=False,
            E01=dict(speedup=1.10, maximum_accuracy_regression=.05, minimum_accuracy_improvement_over_quadratic=1.10),
            E02=dict(error_ratio=1.10, speedup=1.10, minimum_accuracy_improvement_over_quadratic=1.10)),
        adjacent_budget=dict(discretionary_seconds=pilot_seconds+exploration_seconds,
            architecture_seconds=pilot_seconds, protected_exploration_seconds=exploration_seconds,
            exploration_fraction=.20, shared_work_separately_enumerated=True,
            transfer_from_main_campaign=0, monetary_cost=None, energy_joules=None),
        deployment=dict(automatic=False, reason='Require complete-cost margin first'),
        scientific_scope='Smoke/development exploratory; full fresh comparisons require precision, reference, optimization and cost gates')
    units = {}
    def add(name, kind, prior, seconds, **kw):
        units[name] = _unit(name, kind, prior, seconds, **kw)
    add('audit','audit',[],180)
    for name in DIAGNOSTICS:
        deps = ['audit'] if name in ('D01','D05') else ['audit','D01','D05']
        add(name,'diagnostic',deps,180 if smoke else 600, diagnostic=name)
    add('roughness','roughness',['audit'],180 if smoke else 300)
    for name in ('E01','E02'):
        add('explore-'+name,'explore',['audit','D01','D05'],exploration_seconds//2,prototype_ids=[name])
    for track in (['discrete','continuum'] if full else ['both']):
        suffix = '-'+track if full else ''
        add('pilot'+suffix,'train',['audit','D01','D05','D02','D04','roughness'],
            pilot_seconds//2 if full else pilot_seconds, device='cuda', tracks=p['tracks'] if track=='both' else [track])
        add('evaluate'+suffix,'confirm',['pilot'+suffix],1800 if full else 600 if smoke else 1200,
            device='cuda', pilot_unit='pilot'+suffix, tracks=p['tracks'] if track=='both' else [track])
    add('report','report',list(units),600 if full else 240)
    p['units'] = units
    p['stages'] = list(units)
    p['gpu_stages'] = [k for k,v in units.items() if v['device']=='cuda']
    p['budgets'] = {k:{q:v[q] for q in ('seconds','cpus','mem_gib','walltime')} for k,v in units.items()}
    # Imported only on numerical execution; readiness identities themselves are stdlib data.
    p['gpu_test_cases'] = gpu_test_cases()
    p['adjacent_budget']['summed_science_ceiling_seconds'] = sum(v['seconds'] for v in units.values())
    return p

def gpu_test_cases():
    from tdn.analysis.portfolio.protocol import build_protocol as portfolio_protocol
    tracks=('discrete','continuum')
    families=('channel_fixed','channel_global','channel_affine','channel_neural','feature_capacity')
    new = [f'test_adjacent_cuda_limits[{t}-{f}]' for t in tracks for f in families]
    new += [f'test_adjacent_cuda_gradients[{t}-{f}]' for t in tracks for f in ('channel_neural','feature_capacity')]
    new += [f'test_adjacent_cuda_polarization[{t}-{n}]' for t in tracks for n in (2,4)]
    new += [f'test_adjacent_cuda_fitted_control[{f}]' for f in ('channel_global','channel_affine')]
    new += ['test_adjacent_cuda_visible_allocation']
    return new + portfolio_protocol('full')['gpu_test_cases']

def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get('profile') not in PROFILES or digest(protocol)!=digest(build_protocol(protocol['profile'])):
        raise ValueError('Adjacent declaration differs from the immutable profile')
    return protocol

def plan_units(protocol):
    validate_protocol(protocol)
    return list(protocol['units'].values())

def dependencies(protocol, stage):
    return tuple(protocol['units'][stage]['dependencies'])

def resource_for(protocol, stage):
    return dict(protocol['budgets'][stage])
