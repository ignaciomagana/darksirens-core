"""Out-of-catalog host weighting from a Schechter luminosity function."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate

jax.config.update("jax_enable_x64", True)

from darksirens.catalog.completeness import build_observed_density_cache, completion_curves
from darksirens.catalog.hosts import (
    CenteredHostMarks,
    LogLinearHostModel,
    SchechterOutOfCatalogWeight,
    build_marked_incomplete_catalog_prior_state,
    schechter_out_of_catalog_efficiency,
)
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.parameters import CosmologyParameters

COSMO = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
MODEL = LogLinearHostModel(("log_luminosity",), eta_bound=2.5)


def _weight(alpha=-1.2, log_x_min=np.log(0.05), log_x_ref=0.0):
    # a flux limit: x_lim grows as d_L^2, here a smooth stand-in on the grid
    log_x_lim = np.log(0.02) + 2.0 * np.log1p(zgrid / 0.01)
    return SchechterOutOfCatalogWeight(alpha, log_x_min, jnp.asarray(log_x_lim), log_x_ref)


def test_eta_zero_gives_unit_weight():
    got = np.asarray(schechter_out_of_catalog_efficiency(MODEL, jnp.asarray([0.0]), _weight()))
    np.testing.assert_allclose(got, 1.0, rtol=1e-12)


@pytest.mark.parametrize("alpha,eta,x_ref", [(-1.2, 0.7, 1.0), (-0.8, 1.5, 0.3), (-1.5, 0.3, 2.0)])
def test_matches_direct_integration(alpha, eta, x_ref):
    w = _weight(alpha=alpha, log_x_ref=np.log(x_ref))
    got = np.asarray(schechter_out_of_catalog_efficiency(MODEL, jnp.asarray([eta]), w))
    x_min = 0.05
    for i in (50, 300, 700):
        x_lim = float(np.exp(w.log_x_lim[i]))
        if x_lim <= x_min:
            continue
        phi = lambda x: x**alpha * np.exp(-x)
        num = integrate.quad(lambda x: (x / x_ref) ** eta * phi(x), x_min, x_lim, limit=200)[0]
        den = integrate.quad(phi, x_min, x_lim, limit=200)[0]
        np.testing.assert_allclose(got[i], num / den, rtol=1e-8)


def test_empty_interval_takes_the_faint_end_weight():
    w = SchechterOutOfCatalogWeight(-1.2, np.log(0.05), jnp.full(zgrid.shape, np.log(0.01)), 0.0)
    got = np.asarray(schechter_out_of_catalog_efficiency(MODEL, jnp.asarray([1.0]), w))
    np.testing.assert_allclose(got, 0.05, rtol=1e-12)


def test_differentiable_in_eta_and_alpha():
    def f(eta, alpha):
        w = _weight(alpha=alpha)
        return jnp.sum(schechter_out_of_catalog_efficiency(MODEL, jnp.asarray([eta]), w)[100:400])

    g = jax.grad(f, argnums=(0, 1))(0.8, -1.1)
    assert all(np.isfinite(float(x)) for x in g)
    assert float(g[0]) != 0.0 and float(g[1]) != 0.0


def test_prior_state_uses_the_luminosity_function_weight():
    z = np.array([[0.02, 0.03, 0.05, 0.0]])
    cat = GalaxyCatalog(
        apix=1.0e-3,
        zgals=jnp.asarray(z),
        dzgals=jnp.full(z.shape, 1e-3),
        wgals=jnp.asarray([[1.0, 1.0, 1.0, 0.0]]),
        ngals=jnp.asarray([3], dtype=jnp.int32),
    )
    marks = CenteredHostMarks(("log_luminosity",), jnp.asarray([[[0.4], [-0.2], [0.1], [0.0]]]))
    params = CatalogParameters(n0=1.0e-3, delta=0.0, sigma_kde=0.0, z_depth=None)
    cache = build_observed_density_cache(cat)
    w = _weight()
    eta = jnp.asarray([0.9])
    state = build_marked_incomplete_catalog_prior_state(
        COSMO, params, cat, cache, MODEL, marks, eta, out_of_catalog_weight=w
    )
    curves = completion_curves(COSMO, params, cat, cache)
    e_miss = np.asarray(schechter_out_of_catalog_efficiency(MODEL, eta, w))
    np.testing.assert_allclose(np.asarray(state.dN_miss), np.asarray(curves.dN_miss) * e_miss[None, :], rtol=1e-12)
    legacy = build_marked_incomplete_catalog_prior_state(COSMO, params, cat, cache, MODEL, marks, eta)
    np.testing.assert_allclose(np.asarray(state.log_Nobs), np.asarray(legacy.log_Nobs), rtol=0, atol=0)
