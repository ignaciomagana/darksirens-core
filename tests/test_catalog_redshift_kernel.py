import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from darksirens.catalog.redshift import (
    SIGMA_EFF_FLOOR,
    build_catalog_kernel_state,
    eval_log_catalog_prior_state_vmap,
    log_catalog_prior_vmap,
    log_galaxy_measure_grid,
)
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.distances import dV_of_z
from darksirens.cosmology.parameters import CosmologyParameters

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

COSMO = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _catalog(weight_scale=1.0):
    return GalaxyCatalog(
        apix=1.0,
        zgals=jnp.asarray(
            [
                [0.08, 0.20, 0.42, 100.0],
                [0.31, 100.0, 100.0, 100.0],
                [100.0, 100.0, 100.0, 100.0],
            ]
        ),
        dzgals=jnp.asarray(
            [
                [0.01, 0.04, 0.02, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ),
        wgals=jnp.asarray(
            [
                [1.0, 2.0, 0.5, 0.0],
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        * weight_scale,
        ngals=jnp.asarray([3, 1, 0], dtype=jnp.int32),
        unique_pixels=jnp.asarray([7, 42, 9_000_017], dtype=jnp.int32),
    )


def test_galaxy_measure_is_dv_times_one_plus_z_to_delta():
    params = CatalogParameters(delta=0.7, sigma_kde=0.0)
    got = np.exp(np.asarray(log_galaxy_measure_grid(COSMO, params))[1:])
    z = np.asarray(zgrid)[1:]
    truth = np.asarray(
        dV_of_z(zgrid[1:], COSMO.H0, COSMO.Om0, COSMO.w0, COSMO.wa)
    ) * (1.0 + z) ** 0.7
    np.testing.assert_allclose(got, truth, rtol=2e-14, atol=0.0)


def test_sigma_eff_floor_keeps_spectroscopic_galaxy_live():
    state = build_catalog_kernel_state(
        COSMO, CatalogParameters(delta=0.0, sigma_kde=0.0), _catalog()
    )
    assert float(state.sig_eff[1, 0]) == SIGMA_EFF_FLOOR
    assert np.isfinite(float(state.log_kw[1, 0]))


def test_each_occupied_row_is_a_unit_mass_redshift_density():
    cat = _catalog()
    state = build_catalog_kernel_state(
        COSMO, CatalogParameters(delta=0.4, sigma_kde=0.015), cat
    )
    zg = np.asarray(zgrid)
    for row in (0, 1):
        lp = np.asarray(
            eval_log_catalog_prior_state_vmap(
                zgrid,
                jnp.full(zgrid.shape, row, dtype=jnp.int32),
                state,
                cat,
            )
        )
        assert abs(_trapezoid(np.exp(lp), zg) - 1.0) < 3e-3


def test_empty_row_is_exactly_impossible():
    cat = _catalog()
    params = CatalogParameters(delta=0.0, sigma_kde=0.01)
    state = build_catalog_kernel_state(COSMO, params, cat)
    z = jnp.asarray([0.1, 0.3])
    row = jnp.asarray([2, 2], dtype=jnp.int32)
    assert np.all(
        np.isneginf(np.asarray(eval_log_catalog_prior_state_vmap(z, row, state, cat)))
    )
    assert np.all(
        np.isneginf(np.asarray(log_catalog_prior_vmap(z, row, COSMO, params, cat)))
    )


def test_state_and_direct_scalar_kernel_agree_before_underflow_edge():
    cat = _catalog()
    params = CatalogParameters(delta=0.6, sigma_kde=0.02)
    state = build_catalog_kernel_state(COSMO, params, cat)
    z = jnp.asarray([0.03, 0.09, 0.19, 0.32, 0.55, 0.60])
    row = jnp.asarray([0, 0, 0, 1, 0, 1], dtype=jnp.int32)
    direct = np.asarray(log_catalog_prior_vmap(z, row, COSMO, params, cat))
    cached = np.asarray(eval_log_catalog_prior_state_vmap(z, row, state, cat))
    np.testing.assert_allclose(cached, direct, rtol=1e-12, atol=0.0)


def test_state_evaluator_preserves_legacy_far_tail_underflow():
    """The mature cached path is one-pass and floors remote tails at -inf."""

    cat = _catalog()
    params = CatalogParameters(delta=0.6, sigma_kde=0.02)
    state = build_catalog_kernel_state(COSMO, params, cat)
    z = jnp.asarray([1.10])
    row = jnp.asarray([1], dtype=jnp.int32)
    cached = np.asarray(eval_log_catalog_prior_state_vmap(z, row, state, cat))
    direct = np.asarray(log_catalog_prior_vmap(z, row, COSMO, params, cat))
    assert np.isneginf(cached[0])
    assert np.isfinite(direct[0])


def test_base_weight_scale_cancels_from_catalog_shape():
    params = CatalogParameters(delta=0.2, sigma_kde=0.01)
    z = jnp.asarray([0.05, 0.10, 0.20, 0.40])
    row = jnp.zeros(z.shape, dtype=jnp.int32)
    a = np.asarray(log_catalog_prior_vmap(z, row, COSMO, params, _catalog(1.0)))
    b = np.asarray(log_catalog_prior_vmap(z, row, COSMO, params, _catalog(1.0e7)))
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=0.0)


def test_depth_state_is_renormalized_below_depth_and_zero_above():
    cat = _catalog()
    depth = 0.28
    params = CatalogParameters(delta=0.3, sigma_kde=0.02, z_depth=depth)
    state = build_catalog_kernel_state(COSMO, params, cat)
    zg = np.asarray(zgrid)
    lp = np.asarray(
        eval_log_catalog_prior_state_vmap(
            zgrid,
            jnp.zeros(zgrid.shape, dtype=jnp.int32),
            state,
            cat,
        )
    )
    assert np.all(np.isneginf(lp[zg > depth]))
    p = np.where(np.isfinite(lp), np.exp(lp), 0.0)
    assert abs(_trapezoid(p, zg) - 1.0) < 3e-3
    assert float(state.log_depth_mass[0]) < 0.0
