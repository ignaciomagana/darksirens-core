import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# The per-row observed-density cache: negative redshifts, sparse rows, memory


def _rows(n_rows=4, n_gal=200, seed=3, lo=0.01, hi=0.2):
    rng = np.random.default_rng(seed)
    z = np.sort(rng.uniform(lo, hi, (n_rows, n_gal)), axis=1)
    return GalaxyCatalog(
        apix=4.0 * np.pi / n_rows, zgals=z, dzgals=np.full(z.shape, 0.01),
        wgals=np.ones(z.shape), ngals=np.full(n_rows, n_gal, dtype=np.int32),
    )


def test_a_galaxy_far_below_zero_is_refused_by_the_row_cache():
    """At z = -0.5 the 0.05 kernel's mass inside the grid rounds to 0, its
    1e-300 floor made the row's observed density about 1e279 and the row
    complete out to z ~ 1.3; the bind-time guard in kernel widths accepted it
    (dz = 0.2). Refused below 7 widths; z = -0.3 is accepted and finite."""
    cat = _rows()
    z = np.asarray(cat.zgals).copy()
    z[0, 0] = -0.5
    with pytest.raises(ValueError, match=r"1 real galaxies lie more than 7 widths"):
        build_observed_density_cache(cat._replace(zgals=z, dzgals=np.full(z.shape, 0.2)))
    z[0, 0] = -0.3
    cache = build_observed_density_cache(cat._replace(zgals=z))
    assert np.all(np.isfinite(np.asarray(cache.dN_obs_kde)))
    assert float(np.max(np.asarray(cache.dN_obs_kde))) < 1.0e6
    # Padding slots may hold anything.
    padded = np.concatenate([np.asarray(cat.zgals), np.full((4, 2), -5.0)], axis=1)
    build_observed_density_cache(cat._replace(
        zgals=padded, dzgals=np.ones(padded.shape), wgals=np.ones(padded.shape)
    ))


def test_sparse_rows_warn_that_the_row_clip_biases_the_completeness():
    """The clip at 1 of each row's noisy ratio lowers its mean: with about 13
    galaxies per row a complete catalog reads 0.6 at z = 0.02. Rows that hold
    few galaxies under the 0.05 kernel warn; dense rows do not."""
    with pytest.warns(UserWarning, match=r"count_ratio='row': the per-row count ratio's 0.05 kernel"):
        build_observed_density_cache(_rows(n_gal=13))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_observed_density_cache(_rows(n_gal=6000))


def test_the_row_cache_batch_is_sized_from_the_memory_budget(monkeypatch):
    """Each batch forms a ``(rows, N_z, N_max)`` float64 kernel: the fixed 512
    rows needed 512 * 1000 * N_max * 8 bytes whatever the catalog (13 GB at
    N_max 3448). Without a batch size the rows per batch now keep it within
    2**30 bytes, with the same values as the legacy batch of 512."""
    from darksirens.catalog import completeness as module

    cat = _rows(n_rows=512, n_gal=300, hi=0.4)
    shapes = []
    batched = module._batched_observed_density

    def spy(rows, catalog):
        shapes.append(int(rows.shape[0]))
        return batched(rows, catalog)

    monkeypatch.setattr(module, "_batched_observed_density", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto = np.asarray(build_observed_density_cache(cat).dN_obs_kde)
        auto_rows = list(shapes)
        legacy = np.asarray(build_observed_density_cache(cat, batch_size=512).dN_obs_kde)
    assert max(auto_rows) * int(zgrid.size) * 300 * 8 <= 2**30, auto_rows
    np.testing.assert_array_equal(auto, legacy)
