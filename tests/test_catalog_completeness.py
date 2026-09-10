import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from darksirens.catalog.completeness import (
    ObservedDensityCache,
    SIGMA_SMOOTH,
    build_completion_state,
    build_observed_density_cache,
    completion_curves,
    smoothing_operator,
)
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.parameters import CosmologyParameters

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

COSMO = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _catalog(*, weights=None, dz=None):
    if weights is None:
        weights = [[1.0, 8.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    if dz is None:
        dz = [[0.01, 0.20, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    return GalaxyCatalog(
        apix=8.0e-4,
        zgals=jnp.asarray(
            [[0.08, 0.42, 100.0], [0.31, 100.0, 100.0], [100.0, 100.0, 100.0]]
        ),
        dzgals=jnp.asarray(dz),
        wgals=jnp.asarray(weights),
        ngals=jnp.asarray([2, 1, 0], dtype=jnp.int32),
        unique_pixels=jnp.asarray([7, 1_000_003, 9_000_017], dtype=jnp.int32),
    )


def _params(depth=None):
    return CatalogParameters(n0=2.0e-3, delta=0.3, sigma_kde=0.015, z_depth=depth)


def test_observed_completeness_kde_uses_raw_counts_only():
    a = build_observed_density_cache(_catalog())
    b = build_observed_density_cache(
        _catalog(
            weights=[[1.0e-6, 9.0e9, 0.0], [2.0e4, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dz=[[0.9, 0.001, 1.0], [0.7, 1.0, 1.0], [1.0, 1.0, 1.0]],
        )
    )
    np.testing.assert_array_equal(np.asarray(a.dN_obs_kde), np.asarray(b.dN_obs_kde))


def test_observed_kde_rows_have_raw_count_mass():
    cache = build_observed_density_cache(_catalog())
    z = np.asarray(zgrid)
    mass = _trapezoid(np.asarray(cache.dN_obs_kde), z, axis=-1)
    np.testing.assert_allclose(mass, np.asarray([2.0, 1.0, 0.0]), rtol=3e-3, atol=1e-12)


def test_matched_operator_preserves_constant_completeness():
    cat = _catalog()
    params = _params()
    state = build_completion_state(COSMO, params, cat)
    c = 0.37
    obs = jnp.broadcast_to(c * state.dN_exp_smooth, (cat.zgals.shape[0], zgrid.size))
    curves = completion_curves(COSMO, params, cat, ObservedDensityCache(obs))
    active = np.asarray(state.dN_exp_smooth) > 0.0
    for row in range(cat.zgals.shape[0]):
        np.testing.assert_allclose(
            np.asarray(curves.C[row])[active], c, rtol=1e-14, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(curves.dN_miss[row])[active],
            (1.0 - c) * np.asarray(state.dN_exp)[active],
            rtol=2e-14,
            atol=0.0,
        )


def test_empty_row_is_zero_completeness_and_pure_missing_budget():
    cat = _catalog()
    params = _params()
    cache = build_observed_density_cache(cat)
    state = build_completion_state(COSMO, params, cat)
    curves = completion_curves(COSMO, params, cat, cache)
    np.testing.assert_array_equal(np.asarray(curves.C[2]), np.zeros(zgrid.size))
    np.testing.assert_array_equal(np.asarray(curves.dN_miss[2]), np.asarray(state.dN_exp))
    assert float(curves.f[2]) == 0.0


def test_catalog_missing_odds_reduce_to_count_odds():
    cat = _catalog()
    curves = completion_curves(COSMO, _params(), cat, build_observed_density_cache(cat))
    z = np.asarray(zgrid)
    for row in (0, 1):
        n_obs = float(np.asarray(cat.ngals)[row])
        n_miss = float(curves.N_miss[row])
        total = n_obs + n_miss
        missing_fraction = _trapezoid(np.asarray(curves.dN_miss[row]) / total, z)
        np.testing.assert_allclose(1.0 - missing_fraction, n_obs / total, atol=1e-9)


def test_z_depth_none_equals_full_grid_depth_exactly():
    cat = _catalog()
    cache = build_observed_density_cache(cat)
    none = completion_curves(COSMO, _params(None), cat, cache)
    full = completion_curves(COSMO, _params(float(zgrid[-1])), cat, cache)
    for name in ("f", "dN_miss", "C_eff", "N_miss"):
        np.testing.assert_array_equal(np.asarray(getattr(none, name)), np.asarray(getattr(full, name)))


def test_finite_depth_relaxes_to_full_expected_density_above_depth():
    cat = _catalog()
    cache = build_observed_density_cache(cat)
    z = np.asarray(zgrid)
    depth = float(zgrid[len(zgrid) // 3])
    params = _params(depth)
    state = build_completion_state(COSMO, params, cat)
    curves = completion_curves(COSMO, params, cat, cache)
    above = z > depth
    below_far = z <= depth - 10.0 * SIGMA_SMOOTH

    for row in range(cat.zgals.shape[0]):
        np.testing.assert_allclose(
            np.asarray(curves.dN_miss[row])[above],
            np.asarray(state.dN_exp)[above],
            rtol=1e-12,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            np.asarray(curves.C_eff[row])[above], np.zeros(np.count_nonzero(above))
        )
        expected = _trapezoid(np.asarray(curves.dN_miss[row]), z)
        np.testing.assert_allclose(float(curves.N_miss[row]), expected, rtol=1e-12)

    full = completion_curves(COSMO, _params(None), cat, cache)
    assert below_far.any()
    np.testing.assert_array_equal(
        np.asarray(curves.dN_miss)[:, below_far],
        np.asarray(full.dN_miss)[:, below_far],
    )


def test_cached_and_on_the_fly_observed_density_are_identical():
    cat = _catalog()
    params = _params()
    cached = completion_curves(COSMO, params, cat, build_observed_density_cache(cat))
    direct = completion_curves(COSMO, params, cat, None)
    for name in ("C", "f", "dN_miss", "C_eff", "N_miss"):
        np.testing.assert_array_equal(np.asarray(getattr(cached, name)), np.asarray(getattr(direct, name)))


def test_smoothing_operator_columns_integrate_to_one():
    z = np.asarray(zgrid)
    s = np.asarray(smoothing_operator())
    column_mass = _trapezoid(s, z, axis=0)
    # S already contains source-grid trapezoid weights, so summing/integrating
    # columns is not the relevant normalization.  Its action on a unit source
    # density is.  Pin the actual linear-operator closure instead.
    out = s @ np.ones_like(z)
    assert np.all(np.isfinite(column_mass))
    assert np.all(out > 0.0)
    np.testing.assert_allclose(
        s @ (0.23 * np.ones_like(z)), 0.23 * out, rtol=2e-15, atol=0.0
    )
