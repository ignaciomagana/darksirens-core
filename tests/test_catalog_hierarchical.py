"""Focused Phase 5C tests for explicit ordinary/bright hierarchy entry points."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from darksirens.catalog.completeness import build_observed_density_cache
from darksirens.catalog.counterparts import Counterpart
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw.runtime import make_gw_event
from darksirens.likelihood.hierarchical import (
    bright_siren_log_likelihood,
    complete_catalog_siren_log_likelihood,
    dark_siren_log_likelihood,
    spectral_siren_log_likelihood,
)
from darksirens.population import get_fixed_population_params

POP = jnp.asarray(get_fixed_population_params("powerlaw+peak"))
COSMO = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
PARAMS = CatalogParameters(n0=1e-2, delta=0.0, sigma_kde=0.0, z_depth=None)


def _event(n, pixels):
    m1 = np.linspace(34.0, 40.0, n)
    return make_gw_event(
        m1det=m1,
        m2det=0.8 * m1,
        dL=np.linspace(430.0, 530.0, n),
        chieff=np.zeros(n),
        prior_wt=np.ones(n),
        pixels=np.asarray(pixels, dtype=np.int32),
    )


def _catalog(sparse=False):
    zg = np.array([[0.10], [0.10]], dtype=float)
    dz = np.full_like(zg, 0.02)
    wg = np.ones_like(zg)
    ng = np.ones(2, dtype=np.int32)
    if sparse:
        zg[1, 0] = 0.0
        wg[1, 0] = 0.0
        ng[1] = 0
    return GalaxyCatalog(
        apix=np.pi / 3.0,
        zgals=jnp.asarray(zg),
        dzgals=jnp.asarray(dz),
        wgals=jnp.asarray(wg),
        ngals=jnp.asarray(ng),
    )


def test_dark_diagnostics_recompose_total():
    cat = _catalog()
    pe = _event(2, [0, 0])
    sel = _event(16, np.zeros(16, dtype=np.int32))
    cache = build_observed_density_cache(cat)
    d = dark_siren_log_likelihood(
        COSMO,
        PARAMS,
        POP,
        pe,
        cat,
        cache,
        sel,
        cat,
        cache,
        1,
        2,
        16.0,
        pop_model="powerlaw+peak",
        max_likelihood_variance=1e6,
        return_diagnostics=True,
    )
    assembled = d.selection_log_correction + jnp.sum(d.event_log_evidence)
    np.testing.assert_allclose(d.log_likelihood, assembled, rtol=1e-14, atol=0.0)
    assert np.isfinite(float(d.log_likelihood))


def test_complete_empty_policy_is_live():
    cat = _catalog(sparse=True)
    pe = _event(2, [0, 0])
    sel = _event(16, np.tile(np.array([0, 1], dtype=np.int32), 8))
    vals = {}
    for policy in ("zero", "volume"):
        vals[policy] = float(
            complete_catalog_siren_log_likelihood(
                COSMO,
                PARAMS,
                POP,
                pe,
                cat,
                sel,
                cat,
                1,
                2,
                16.0,
                pop_model="powerlaw+peak",
                empty_policy=policy,
                max_likelihood_variance=1e6,
            )
        )
    assert np.isfinite(vals["zero"])
    assert np.isfinite(vals["volume"])
    assert vals["zero"] != vals["volume"]

    default = float(
        complete_catalog_siren_log_likelihood(
            COSMO,
            PARAMS,
            POP,
            pe,
            cat,
            sel,
            cat,
            1,
            2,
            16.0,
            pop_model="powerlaw+peak",
            max_likelihood_variance=1e6,
        )
    )
    assert default == vals["zero"]


def test_bright_selection_is_exactly_catalog_free_volume_selection():
    cat = _catalog()
    pe = _event(2, [0, 0])
    sel = _event(16, np.zeros(16, dtype=np.int32))
    bright = bright_siren_log_likelihood(
        COSMO,
        POP,
        pe,
        cat,
        (Counterpart(z=0.10, dz=0.02, pixel=0, sky_marginalized=False),),
        sel,
        1,
        2,
        16.0,
        pop_model="powerlaw+peak",
        max_likelihood_variance=1e6,
        return_diagnostics=True,
    )
    spectral = spectral_siren_log_likelihood(
        COSMO,
        POP,
        pe,
        sel,
        1,
        2,
        16.0,
        pop_model="powerlaw+peak",
        max_likelihood_variance=1e6,
        return_diagnostics=True,
    )
    assert float(bright.log_mu) == float(spectral.log_mu)
    assert float(bright.n_eff) == float(spectral.n_eff)
    assert float(bright.selection_log_correction) == float(spectral.selection_log_correction)
