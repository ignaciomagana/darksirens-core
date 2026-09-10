"""Catalog-free comoving-volume redshift prior used by spectral sirens."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.cosmology.volume import (
    log_comoving_volume_prior,
    normalized_comoving_volume_grid,
)


def _cosmo(h0):
    return CosmologyParameters(H0=h0, Om0=0.3075, w0=-1.0, wa=0.0)


def test_volume_grid_normalizes_to_one():
    p = normalized_comoving_volume_grid(_cosmo(67.74))
    np.testing.assert_allclose(float(jnp.trapezoid(p, zgrid)), 1.0, rtol=0, atol=2e-15)
    assert float(p[0]) == 0.0
    assert np.all(np.asarray(p[1:]) > 0.0)


def test_normalized_volume_prior_is_h0_scale_invariant():
    z = jnp.asarray([1e-3, 0.03, 0.1, 0.3, 1.0, 3.0])
    a = log_comoving_volume_prior(z, _cosmo(50.0))
    b = log_comoving_volume_prior(z, _cosmo(100.0))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=0)


def test_low_z_log_prior_has_quadratic_limit_and_finite_gradient():
    cosmo = _cosmo(67.74)
    z1 = 4e-4
    z2 = 8e-4
    lp1 = float(log_comoving_volume_prior(jnp.asarray(z1), cosmo))
    lp2 = float(log_comoving_volume_prior(jnp.asarray(z2), cosmo))
    np.testing.assert_allclose(lp2 - lp1, 2.0 * np.log(z2 / z1), rtol=0, atol=2e-12)

    g = jax.grad(lambda h: log_comoving_volume_prior(jnp.asarray(0.1), _cosmo(h)))(67.74)
    assert np.isfinite(float(g))
