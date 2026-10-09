"""Exact draw preservation and bounded primary inference."""
import numpy as np
import pytest
from tdn.analysis.roadmap.statistics import paired_cluster_bootstrap
from tdn.analysis.portfolio.statistics import paired_cluster_interval_fast,field_cluster_interval


@pytest.mark.parametrize('fields,seeds',[(0,1),(1,3),(2,2),(3,3),(5,2),(24,3),(100,5),(300,10)])
@pytest.mark.parametrize('repeats',[1,10,1000])
def test_vectorized_paired_bootstrap_preserves_legacy_draws(fields,seeds,repeats):
    rows=[dict(field_cluster=f'f{f}',seed=s,difference=float(np.sin(f+.7*s)+.13*q))
          for f in range(fields) for s in range(seeds) for q in range(3)]
    old=paired_cluster_bootstrap(rows,repeats=repeats)
    new=paired_cluster_interval_fast(rows,repeats=repeats)
    assert old.keys()==new.keys()
    for key in old:
        if isinstance(old[key],float):assert old[key]==pytest.approx(new[key],abs=1e-14,rel=0.)
        else:assert old[key]==new[key]


def test_vectorized_interval_rejects_missing_cells_and_preserves_repeated_unit_count():
    rows=[dict(field_cluster=f'f{f}',seed=s,difference=float(f+.1*s)) for f in range(8) for s in range(3)]
    assert field_cluster_interval(rows[:-1])['status']=='INCOMPLETE_CROSSED_DESIGN'
    one=field_cluster_interval(rows);seven=field_cluster_interval(rows*7)
    assert one['independent_fields']==seven['independent_fields']==8
    assert one['lower']==pytest.approx(seven['lower'])
    assert one['upper']==pytest.approx(seven['upper'])
    assert field_cluster_interval(rows[:12],minimum_fields=5)['status']=='INSUFFICIENT_INDEPENDENT_FIELDS'


def test_vectorized_interval_rejects_nonfinite_observations():
    with pytest.raises(ValueError,match='finite'):
        field_cluster_interval([dict(field_cluster='x',seed=1,difference=float('nan'))])
