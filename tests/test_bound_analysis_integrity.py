"""A ``BoundAnalysis`` evaluates only the likelihood its analysis describes.

The likelihood picks the count-ratio estimator by the type of the cache it is
given, reads a row fraction whenever its operands carry one, runs the float64
program on whatever columns it holds and traces its programs lazily under the
current global settings. A binding rebuilt with ``dataclasses.replace`` (or
unpickled) is checked against its analysis like one ``bind_analysis`` makes,
its programs are traced under the settings it was bound with, and
``core_numerics_semantic(bound)`` reports those. The resolved soft guard is a
bool, and a pooled record built directly takes the default window.
"""

from __future__ import annotations

import dataclasses
import pickle
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import darksirens as ds
from darksirens.analysis import IncompleteCatalogRedshift
from darksirens.catalog.completeness import (
    POOLED_WINDOW_DEFAULT,
    build_observed_density_cache,
)
from darksirens.inference.run_fingerprint import core_numerics_semantic
from darksirens.population.utils import (
    configure_normalization_grids,
    normalization_grid_settings,
)
from darksirens.runtime_binding import bind_analysis

from _historical_settings import catalog_settings
from test_count_ratio_pooled import (
    A,
    A_DEEP,
    B,
    EVENTS,
    INJECTIONS,
    SEL,
    _bind,
    _model,
    _points,
    _pooled,
)


def _quiet_bind(analysis, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="count_ratio='pooled'")
        return _bind(analysis, **kwargs)


# ---------------------------------------------------------------------------
# Operands derived from the analysis


def test_a_replaced_analysis_with_another_count_ratio_is_refused():
    pooled = _quiet_bind(_pooled(A))
    row = _quiet_bind(_model(A, count_ratio="row"))
    with pytest.raises(ValueError, match="needs ObservedDensityCache .*carries PooledCountRatioCache"):
        dataclasses.replace(pooled, analysis=row.analysis)
    with pytest.raises(ValueError, match="needs PooledCountRatioCache .*carries ObservedDensityCache"):
        dataclasses.replace(row, analysis=pooled.analysis)
    # completeness="selection" reads no cache.
    selection = _model(A, completeness="selection", selection=SEL)
    with pytest.raises(ValueError, match="needs no cache"):
        dataclasses.replace(pooled, analysis=selection)
    # The same analysis, rebuilt, is accepted and evaluates the same.
    theta = _points(pooled.analysis, 1)[0]
    assert float(dataclasses.replace(pooled)(theta)) == float(pooled(theta))


def test_a_pooled_cache_of_another_depth_is_refused():
    deep = _quiet_bind(_pooled(A_DEEP))
    assert deep.z_depth == 0.3 and deep.observed_density_cache.depth.z_depth == 0.3
    with pytest.raises(ValueError, match="built for z_depth=0.3, but the binding evaluates z_depth=None"):
        dataclasses.replace(deep, z_depth=None)
    shallow = _quiet_bind(_pooled(A))
    with pytest.raises(ValueError, match="built for z_depth=None, but the binding evaluates z_depth=0.3"):
        dataclasses.replace(shallow, z_depth=0.3)


def test_the_selection_row_fraction_must_match_the_analysis():
    fraction = np.linspace(0.2, 1.0, 48)
    with_fraction = _bind(_model(A, completeness="selection", selection=SEL, row_fraction=fraction))
    without = _bind(_model(A, completeness="selection", selection=SEL))
    with pytest.raises(ValueError, match="has a completeness='selection' row fraction, but this BoundAnalysis carries none"):
        dataclasses.replace(with_fraction, model_operands=None)
    with pytest.raises(ValueError, match="reads no model operands"):
        dataclasses.replace(without, model_operands=with_fraction.model_operands)
    with pytest.raises(ValueError, match="reads no model operands"):
        dataclasses.replace(with_fraction, analysis=without.analysis)


def test_float32_columns_need_compute_dtype_float32():
    analysis = _model(A, count_ratio="row")
    single = _bind(analysis, compute_dtype="float32")
    assert single.gw_pe.dL.dtype == jnp.float32
    with pytest.raises(ValueError, match="gw_pe.m1det is float32, but this BoundAnalysis has compute_dtype=None"):
        dataclasses.replace(single, compute_dtype=None)


def test_a_field_mixture_component_cache_must_match_the_analysis():
    pooled = _quiet_bind(_pooled([A_DEEP, B], catalog_sky_weighting="field"))
    row = _quiet_bind(_model([A_DEEP, B], catalog_sky_weighting="field", count_ratio="row"))
    with pytest.raises(ValueError, match="catalog 1 .*compact view needs ObservedDensityCache"):
        dataclasses.replace(pooled, analysis=row.analysis)
    first, second = pooled.model_operands.components
    swapped = pooled.model_operands._replace(components=(
        first._replace(compact_cache=build_observed_density_cache(first.compact)), second,
    ))
    with pytest.raises(ValueError, match="catalog 1 .*compact view needs PooledCountRatioCache"):
        dataclasses.replace(pooled, model_operands=swapped)
    wrong_depth = pooled.model_operands._replace(components=(
        first._replace(full_cache=second.full_cache), second,
    ))
    with pytest.raises(ValueError, match="catalog 1 .*full view: the pooled count-ratio cache was built for z_depth=None"):
        dataclasses.replace(pooled, model_operands=wrong_depth)
    theta = _points(pooled.analysis, 1)[0]
    assert float(dataclasses.replace(pooled)(theta)) == float(pooled(theta))


def test_an_unpickled_binding_is_checked_and_evaluates_the_same():
    bound = _quiet_bind(_pooled(A_DEEP))
    theta = _points(bound.analysis, 1)[0]
    clone = pickle.loads(pickle.dumps(bound))
    assert float(clone(theta)) == float(bound(theta))


# ---------------------------------------------------------------------------
# The settings a trace reads


@pytest.fixture
def restore_grids():
    before = normalization_grid_settings()
    yield before
    configure_normalization_grids(
        n_mass=before.n_mass, n_q=before.n_q, n_chi=before.n_chi,
        pairing_norm=before.pairing_norm, pairing_scale=before.pairing_scale,
        pairing_m1_grid=before.pairing_m1_grid,
    )
    assert normalization_grid_settings() == before


def test_a_trace_after_the_grid_settings_changed_is_refused(restore_grids):
    """The diagnostics program (guard report) is traced at its first call. Had
    the grids changed since binding, it evaluated another likelihood than the
    one sampled, and the fingerprint of the binding reported the new grids."""
    analysis = ds.model(cosmology=ds.Cosmology(H0=(50.0, 90.0)), population=ds.Population(
        "powerlaw+peak", fixed=True))
    bound = bind_analysis(analysis, events=EVENTS, injections=INJECTIONS)
    theta = np.array([67.0])
    value = float(bound(theta))
    recorded = core_numerics_semantic(bound)
    n_mass = restore_grids.n_mass
    configure_normalization_grids(n_mass=60, pairing_norm="per_sample")
    # The compiled program is unchanged and keeps serving calls.
    assert float(bound(theta)) == value
    with pytest.raises(RuntimeError, match=f"n_mass {n_mass} -> 60.*pairing_norm 'auto' -> 'per_sample'"):
        bound.diagnostics(theta)
    assert core_numerics_semantic(bound) == recorded
    assert core_numerics_semantic(bound)["normalization_grids"]["n_mass"] == n_mass
    assert core_numerics_semantic()["normalization_grids"]["n_mass"] == 60
    # A binding made now traces under the new settings.
    assert np.isfinite(float(bind_analysis(analysis, events=EVENTS, injections=INJECTIONS)(theta)))
    configure_normalization_grids(n_mass=n_mass, pairing_norm=restore_grids.pairing_norm)
    assert float(bound.diagnostics(theta).log_likelihood) == pytest.approx(value, rel=1e-12)


def test_a_trace_after_the_missing_density_form_changed_is_refused():
    bound = _quiet_bind(_pooled(A_DEEP))
    theta = _points(bound.analysis, 1)[0]
    before = bound._trace_settings[1].missing_density
    other = "grid" if before != "grid" else "gather"
    with catalog_settings(missing_density=other):
        with pytest.raises(RuntimeError, match=f"missing_density {before!r} -> {other!r}"):
            bound(theta)
        # A kernel layout or window change does not enter a trace: those are
        # fixed in the binding's catalog views.
    with catalog_settings(kernel_layout="padded", kernel_window="off"):
        assert np.isfinite(float(bound(theta)))


# ---------------------------------------------------------------------------
# The resolved soft guard and a directly built pooled record


@pytest.mark.parametrize("mode", ["hard", "off", "False", "soft", 1, None])
def test_the_soft_guard_flag_must_be_a_bool(mode):
    analysis = ds.model(population=ds.Population("powerlaw+peak", fixed=True))
    with pytest.raises(TypeError, match="selection_neff_soft_guard must be a bool"):
        bind_analysis(analysis, events=EVENTS, injections=INJECTIONS,
                      selection_neff_soft_guard=mode)


def test_a_numpy_bool_soft_guard_is_accepted_and_replace_is_checked():
    analysis = ds.model(population=ds.Population("powerlaw+peak", fixed=True))
    bound = bind_analysis(analysis, events=EVENTS, injections=INJECTIONS,
                          selection_neff_soft_guard=np.True_)
    assert bound.selection_neff_soft_guard is True
    with pytest.raises(TypeError, match="got str 'hard'"):
        dataclasses.replace(bound, selection_neff_soft_guard="hard")


@pytest.mark.parametrize("weighting", ["conditional", "field"])
def test_a_direct_pooled_record_without_a_window_takes_the_default(weighting):
    """ds.model resolves the window; a record built directly with
    ``count_ratio="pooled"`` and no window used to fail on float(None)."""
    analysis = _model(A, catalog_sky_weighting=weighting)
    assert analysis.redshift.count_ratio == "pooled"
    assert analysis.redshift.count_ratio_window == POOLED_WINDOW_DEFAULT
    if weighting == "conditional":
        record = IncompleteCatalogRedshift(
            A, n0_units=analysis.redshift.n0_units, count_ratio="pooled"
        )
    else:
        record = dataclasses.replace(analysis.redshift, count_ratio_window=None)
    assert record.count_ratio_window is None
    direct = dataclasses.replace(analysis, redshift=record)
    want, got = _quiet_bind(analysis), _quiet_bind(direct)
    for theta in _points(analysis):
        assert float(got(theta)) == float(want(theta))
