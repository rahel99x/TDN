"""Display-only interpolation, with raw observations and missing intervals retained.

None of these helpers is used by a solver, model selection or statistical test.
PCHIP is a drawing convention, not a fitted scientific response function.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np
from scipy.interpolate import PchipInterpolator


def _number(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def smooth_curve(x, y, *, lower=None, upper=None, points_per_interval=12,
                 expected_step=None, positive=False):
    """Return independent finite display segments; never cross an explicit gap.

    Missing x cannot locate an observation and is reported in ``unlocated``.
    Duplicated x must be explicitly summarized by the caller, not silently
    averaged. Ranges are observed minima/maxima, never confidence intervals.
    """
    if len(x) != len(y) or (lower is None) != (upper is None):
        raise ValueError('Curve and range arrays must have matching lengths')
    has_range = lower is not None
    if has_range and (len(lower) != len(x) or len(upper) != len(x)):
        raise ValueError('Curve and range arrays must have matching lengths')
    if not 2 <= points_per_interval <= 100:
        raise ValueError('Display sampling must be bounded between 2 and 100')
    if expected_step is not None and (not _number(expected_step) or expected_step <= 0):
        raise ValueError('Expected spacing must be positive')
    rows, unlocated = [], []
    for index, (a, b) in enumerate(zip(x, y)):
        record = {'x': a, 'y': b, 'index': index}
        if has_range:
            record.update(lower=lower[index], upper=upper[index])
        if not _number(a):
            unlocated.append(record)
        else:
            rows.append(record)
    rows.sort(key=lambda row: row['x'])
    if any(a['x'] == b['x'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Duplicate abscissae require an explicit scientific aggregation')
    groups, current, gaps = [], [], []
    previous = None
    for row in rows:
        valid = _number(row['y']) and (not positive or row['y'] > 0)
        if has_range:
            valid = valid and _number(row['lower']) and _number(row['upper'])
            if valid and not row['lower'] <= row['y'] <= row['upper']:
                raise ValueError('Observed range must contain its median')
            valid = valid and (not positive or row['lower'] > 0)
        omitted = previous is not None and expected_step is not None and row['x']-previous > 1.5*expected_step
        if omitted or not valid:
            if current: groups.append(current)
            current = []
            gaps.append({'x': row['x'], 'reason': 'unobserved interval' if omitted else 'missing/nonpositive observation'})
        if valid:
            current.append(row)
        previous = row['x']
    if current: groups.append(current)
    segments = []
    for group in groups:
        xs = np.asarray([row['x'] for row in group], dtype=float)
        keys = ['y', 'lower', 'upper'] if has_range else ['y']
        raw = {key: np.asarray([row[key] for row in group], dtype=float) for key in keys}
        dense = (np.concatenate([np.linspace(a, b, points_per_interval, endpoint=False)
                    for a, b in zip(xs, xs[1:])] + [xs[-1:]]) if len(xs) > 1 else xs)
        method = 'PCHIP display interpolation' if len(xs) >= 3 else 'linear display interpolation' if len(xs) == 2 else 'single observation'
        values = {key: (PchipInterpolator(xs, val)(dense) if len(xs) >= 3 else np.interp(dense, xs, val))
                  for key, val in raw.items()}
        # Independently shape-preserving curves can still cross one another.
        # Linear interpolation of ordered endpoint triples preserves the order.
        if has_range and (np.any(values['lower'] > values['y']) or np.any(values['y'] > values['upper'])):
            values = {key: np.interp(dense, xs, val) for key, val in raw.items()}
            method = 'linear display interpolation (PCHIP envelope crossing avoided)'
        segments.append({'x': dense.tolist(), **{key: val.tolist() for key, val in values.items()},
                         'observed': group, 'method': method})
    return {'segments': segments, 'gaps': gaps, 'unlocated': unlocated,
            'meaning': 'Display interpolation only; observed min/max ranges are not confidence intervals.'}


def observed_windows(rows, xkey, ykey, *, maximum_windows=40):
    """Summarize a single declared series into equally spaced display windows.

    Raw values and positions remain in each window. Empty windows are explicit.
    These are display summaries, never data for a scientific comparison.
    """
    positioned = [r for r in rows if _number(r.get(xkey))]
    if not positioned:
        return []
    xs = sorted(set(float(r[xkey]) for r in positioned))
    count = min(maximum_windows, len(xs))
    if len(xs) <= maximum_windows:
        buckets = [[r for r in positioned if float(r[xkey]) == x] for x in xs]
        centers = xs
    else:
        edges = np.linspace(xs[0], xs[-1], count+1)
        buckets = [[] for _ in range(count)]
        centers = ((edges[:-1]+edges[1:])/2).tolist()
        for row in positioned:
            index = min(count-1, int((float(row[xkey])-xs[0])/(xs[-1]-xs[0])*count))
            buckets[index].append(row)
    result = []
    for center, bucket in zip(centers, buckets):
        values = [float(r[ykey]) for r in bucket if _number(r.get(ykey)) and r[ykey] > 0]
        result.append({'x': center, 'median': float(np.median(values)) if values else None,
            'lower': min(values) if values else None, 'upper': max(values) if values else None,
            'observation_count': len(values), 'invalid_or_nonpositive_count': len(bucket)-len(values),
            'raw': [{'x': r[xkey], 'y': r.get(ykey), 'source': r.get('_source')} for r in bucket]})
    return result


def draw_curve(ax, curve, *, style, label=None, show_points=True, observed_range=False):
    """Draw thin continuous segments and visible original observations."""
    for index, segment in enumerate(curve['segments']):
        kwargs = {key: style[key] for key in ('color', 'linestyle', 'linewidth') if key in style}
        ax.plot(segment['x'], segment['y'], label=label if index == 0 else None, **kwargs)
        if observed_range and 'lower' in segment:
            ax.fill_between(segment['x'], segment['lower'], segment['upper'], color=style.get('color'), alpha=.09)
            for key in ('lower', 'upper'):
                ax.plot(segment['x'], segment[key], color=style.get('color'), linewidth=.45, alpha=.85)
        if show_points:
            ax.scatter([r['x'] for r in segment['observed']], [r['y'] for r in segment['observed']],
                marker=style.get('marker', 'o'), s=13, facecolors='none', edgecolors=style.get('color'), linewidths=.55)


def spatial_display_metadata(array, display='raw', domain=(1., 1.)):
    """Describe raster drawing without changing values, extrema or resolution."""
    values = np.asarray(array)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError('Spatial display requires a real two-dimensional array')
    if display not in ('raw', 'bicubic') or len(domain) != 2 or any(not _number(x) or x <= 0 for x in domain):
        raise ValueError('Unsupported spatial display or physical domain')
    ny, nx = values.shape
    if min(nx, ny) < 1:
        raise ValueError('Spatial display requires nonempty axes')
    dx, dy = domain[0]/nx, domain[1]/ny
    return {'display': display, 'interpolation': 'nearest' if display == 'raw' else 'bicubic',
        'shape': list(values.shape), 'dtype': str(values.dtype),
        'array_sha256': hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
        'domain': list(domain), 'extent': [-dx/2, domain[0]-dx/2, -dy/2, domain[1]-dy/2],
        'origin': 'lower', 'interpolation_stage': 'data',
        'label': 'Raw grid samples (nearest display)' if display == 'raw' else 'Bicubic display only; original grid and metrics unchanged',
        'warning': 'Interpolation may overshoot between samples; display colors clip to the same raw-data limits. No interpolated extrema enter metrics.'}


def spatial_image(ax, array, *, display='raw', domain=(1., 1.), vmin=None, vmax=None, cmap='viridis'):
    metadata = spatial_display_metadata(array, display, domain)
    image = ax.imshow(array, origin=metadata['origin'], extent=metadata['extent'],
        interpolation=metadata['interpolation'], interpolation_stage='data',
        vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
    metadata.update(vmin=image.norm.vmin, vmax=image.norm.vmax, cmap=cmap)
    return image, metadata
