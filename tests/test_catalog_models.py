import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from darksirens.catalog.completeness import build_observed_density_cache
from darksirens.catalog.counterparts import (
    Counterpart,
    build_counterpart_prior_state,
    eval_counterpart_prior_state_vmap,
)
from darksirens.catalog.models import (
    build_complete_catalog_prior_state,
    build_incomplete_catalog_prior_state,
    eval_complete_catalog_prior_state,
    eval_complete_catalog_prior_state_vmap,
    eval_incomplete_catalog_prior_state_vmap,
)
from darksirens.catalog.redshift import eval_log_catalog_prior_state_vmap
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.cosmology.volume import log_comoving_volume_prior


_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _cosmo():
    return CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _params(z_depth=None):
    return CatalogParameters(n0=2.0e-3, delta=0.4, sigma_kde=0.01, z_depth=z_depth)


def _catalog():
    return GalaxyCatalog(
        apix=8.0e-4,
        zgals=jnp.asarray([
            [0.08, 0.21, 100.0],
            [100.0, 100.0, 100.0],
            [0.34, 100.0, 100.0],
        ]),
        dzgals=jnp.asarray([
            [0.01, 0.03, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]),
        wgals=jnp.asarray([
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]),
        ngals=jnp.asarray([2, 0, 1], dtype=jnp.int32),
        unique_pixels=jnp.asarray([7, 42, 1_000_003], dtype=jnp.int32),
    )


def test_incomplete_prior_is_normalized_and_empty_row_is_pure_missing():
    cat = _catalog()
    cache = build_observed_density_cache(cat)
    state = build_incomplete_catalog_prior_state(_cosmo(), _params(), cat, cache)
    z = zgrid
    for row in range(3):
        lp = np.asarray(eval_incomplete_catalog_prior_state_vmap(
            z, jnp.full(z.shape, row, dtype=jnp.int32), state, cat
        ))
        np.testing.assert_allclose(_trapezoid(np.exp(lp), np.asarray(z)), 1.0, rtol=5e-3)

    # Empty row: no observed host amplitude, so the prior is exactly the
    # normalized missing-density branch at grid nodes.
    row = 1
    p = np.exp(np.asarray(eval_incomplete_catalog_prior_state_vmap(
        z, jnp.full(z.shape, row, dtype=jnp.int32), state, cat
    )))
    expected = np.asarray(state.dN_miss[row]) / np.exp(float(state.log_Z[row]))
    np.testing.assert_allclose(p[1:], expected[1:], rtol=1e-12, atol=0.0)


def test_incomplete_depth_zeros_observed_catalog_branch_above_depth():
    cat = _catalog()
    params = _params(z_depth=0.25)
    cache = build_observed_density_cache(cat)
    state = build_incomplete_catalog_prior_state(_cosmo(), params, cat, cache)
    assert np.all(np.isneginf(np.asarray(
        eval_log_catalog_prior_state_vmap(
            zgrid[zgrid > 0.25],
            jnp.zeros(int(np.sum(np.asarray(zgrid > 0.25))), dtype=jnp.int32),
            state.kernels,
            cat,
        )
    )))
    # The total prior remains finite there through the missing branch.
    z_hi = jnp.asarray([0.4])
    lp = eval_incomplete_catalog_prior_state_vmap(
        z_hi, jnp.asarray([0], dtype=jnp.int32), state, cat
    )
    assert np.isfinite(float(lp[0]))


def test_complete_empty_policies_and_occupied_row():
    cat = _catalog()
    state = build_complete_catalog_prior_state(_cosmo(), _params(), cat)
    z = jnp.asarray([0.12, 0.12, 0.34])
    rows = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    zero = np.asarray(eval_complete_catalog_prior_state_vmap(
        z, rows, state, cat, empty_policy="zero"
    ))
    volume = np.asarray(eval_complete_catalog_prior_state_vmap(
        z, rows, state, cat, empty_policy="volume"
    ))
    assert np.isfinite(zero[0]) and np.isfinite(zero[2])
    assert np.isneginf(zero[1])
    assert np.isfinite(volume[1])
    np.testing.assert_array_equal(zero[[0, 2]], volume[[0, 2]])

    expected_volume = float(log_comoving_volume_prior(jnp.asarray(0.12), _cosmo()))
    np.testing.assert_allclose(volume[1], expected_volume, rtol=1e-12, atol=0.0)


def test_complete_empty_policy_defaults_to_the_strict_branch():
    cat = _catalog()
    state = build_complete_catalog_prior_state(_cosmo(), _params(), cat)
    z = jnp.asarray([0.12, 0.12, 0.34])
    rows = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    default = np.asarray(eval_complete_catalog_prior_state_vmap(z, rows, state, cat))
    zero = np.asarray(eval_complete_catalog_prior_state_vmap(
        z, rows, state, cat, empty_policy="zero"
    ))
    np.testing.assert_array_equal(default, zero)
    assert np.isneginf(default[1])

    scalar = float(eval_complete_catalog_prior_state(
        jnp.asarray(0.12), jnp.asarray(1, dtype=jnp.int32), state, cat
    ))
    assert np.isneginf(scalar)


def test_complete_invalid_empty_policy_fails_loudly():
    cat = _catalog()
    state = build_complete_catalog_prior_state(_cosmo(), _params(), cat)
    try:
        eval_complete_catalog_prior_state_vmap(
            jnp.asarray([0.1]), jnp.asarray([1], dtype=jnp.int32), state, cat,
            empty_policy="guess",
        )
    except ValueError as exc:
        assert "zero" in str(exc) and "volume" in str(exc)
    else:
        raise AssertionError("invalid empty policy did not raise")


def test_counterpart_prior_multiplies_volume_factor_and_gates_global_pixel():
    cat = _catalog()
    cp = Counterpart(z=0.10, dz=0.02, pixel=7, sky_marginalized=False)
    state = build_counterpart_prior_state(_cosmo(), cp)
    z = jnp.asarray([0.08, 0.12, 0.08])
    rows = jnp.asarray([0, 0, 1], dtype=jnp.int32)
    got = np.asarray(eval_counterpart_prior_state_vmap(z, rows, state, cat))

    from jax.scipy.stats import norm

    expected = np.asarray(
        norm.logpdf(z[:2], cp.z, cp.dz)
        + jax.vmap(lambda zz: log_comoving_volume_prior(zz, _cosmo()))(z[:2])
    )
    np.testing.assert_allclose(got[:2], expected, rtol=1e-12, atol=0.0)
    assert np.isneginf(got[2])


def test_counterpart_sky_marginalization_removes_pixel_gate():
    cat = _catalog()
    cp = Counterpart(z=0.10, dz=0.02, pixel=7, sky_marginalized=True)
    state = build_counterpart_prior_state(_cosmo(), cp)
    got = np.asarray(eval_counterpart_prior_state_vmap(
        jnp.asarray([0.10, 0.10]),
        jnp.asarray([0, 1], dtype=jnp.int32),
        state,
        cat,
    ))
    assert np.all(np.isfinite(got))
    np.testing.assert_array_equal(got[:1], got[1:])
