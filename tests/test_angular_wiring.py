from __future__ import annotations

import numpy as np
import pytest

import darksirens as ds

jnp = pytest.importorskip("jax.numpy")

from darksirens.catalog.completeness import build_observed_density_cache
from darksirens.catalog.io import CatalogStore
from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
from darksirens.cosmology.parameters import CosmologyParameters
from darksirens.gw.runtime import make_gw_event
from darksirens.likelihood.hierarchical import (
    dark_siren_log_likelihood,
    spectral_siren_log_likelihood,
)
from darksirens.population import get_fixed_population_params
from darksirens.runtime_binding import _decode_theta


POP = jnp.asarray(get_fixed_population_params("powerlaw+peak"))
COSMO = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
PARAMS = CatalogParameters(n0=1e-2, delta=0.0, sigma_kde=0.0, z_depth=None)


def _directions(n):
    phi = np.linspace(0.1, 5.8, n)
    z = np.linspace(-0.75, 0.75, n)
    r = np.sqrt(1.0 - z * z)
    return r * np.cos(phi), r * np.sin(phi), z


def _event(n, pixels):
    m1 = np.linspace(34.0, 40.0, n)
    nx, ny, nz = _directions(n)
    return make_gw_event(
        m1det=m1,
        m2det=0.8 * m1,
        dL=np.linspace(430.0, 530.0, n),
        chieff=np.zeros(n),
        prior_wt=np.ones(n),
        pixels=np.asarray(pixels, dtype=np.int32),
        nx=nx,
        ny=ny,
        nz=nz,
    )


def _catalog_store():
    cat = GalaxyCatalog(
        apix=np.pi / 3.0,
        zgals=np.array([[0.10, 100.0]]),
        dzgals=np.array([[0.02, 1.0]]),
        wgals=np.array([[1.0, 0.0]]),
        ngals=np.array([1], dtype=np.int32),
        unique_pixels=None,
    )
    return CatalogStore(path="memory.h5", nside=1, z_depth=0.3, catalog=cat)


def _runtime_catalog():
    store = _catalog_store()
    cat = store.catalog
    return GalaxyCatalog(
        apix=jnp.asarray(cat.apix),
        zgals=jnp.asarray(cat.zgals),
        dzgals=jnp.asarray(cat.dzgals),
        wgals=jnp.asarray(cat.wgals),
        ngals=jnp.asarray(cat.ngals),
    )


def test_public_plan_appends_dipole_after_catalog_and_resolves_ball3():
    analysis = ds.model(
        cosmology=ds.Cosmology(H0=67.74, Om0=0.3075),
        population=ds.Population("powerlaw+peak", fixed=True),
        catalog=_catalog_store(),
        angular="dipole",
    )
    assert analysis.angular_model == "dipole"
    assert analysis.parameters.labels == (
        "log10n0",
        "delta",
        "sigma_kde",
        r"$d_x$",
        r"$d_y$",
        r"$d_z$",
    )
    assert analysis.parameters.n_catalog == 3
    assert analysis.parameters.n_angular == 3
    assert analysis.parameters.angular_labels == (r"$d_x$", r"$d_y$", r"$d_z$")
    assert analysis.parameters.joint_constraints == (("ball3", (3, 4, 5)),)

    theta = jnp.asarray([-2.0, 0.0, 0.01, 0.2, -0.1, 0.05])
    _cosmo, _pop, _catalog, angular = _decode_theta(
        analysis, theta, z_depth=0.3
    )
    np.testing.assert_array_equal(np.asarray(angular), np.asarray(theta[-3:]))


def test_isotropic_plan_is_exact_empty_angular_block():
    analysis = ds.model(
        cosmology=ds.Cosmology(H0=67.74, Om0=0.3075),
        population=ds.Population("powerlaw+peak", fixed=True),
    )
    assert analysis.angular_model == "isotropic"
    assert analysis.parameters.n_angular == 0
    assert analysis.parameters.angular_labels == ()
    assert analysis.parameters.labels == ()
    assert analysis.parameters.joint_constraints == ()


def test_unknown_angular_model_fails_at_construction():
    with pytest.raises(ValueError, match="Unknown angular model"):
        ds.model(
            cosmology=ds.Cosmology(H0=67.74, Om0=0.3075),
            population=ds.Population("powerlaw+peak", fixed=True),
            angular="not-a-sky-model",
        )


def test_spectral_explicit_isotropic_is_bit_identical_to_accepted_path():
    pe = _event(6, np.zeros(6, dtype=np.int32))
    sel = _event(64, np.zeros(64, dtype=np.int32))
    common = dict(
        pop_model="powerlaw+peak",
        max_likelihood_variance=1e6,
        return_diagnostics=True,
    )
    baseline = spectral_siren_log_likelihood(
        COSMO, POP, pe, sel, 1, 6, 128.0, **common
    )
    isotropic = spectral_siren_log_likelihood(
        COSMO,
        POP,
        pe,
        sel,
        1,
        6,
        128.0,
        angular_model="isotropic",
        angular_params=jnp.asarray([]),
        **common,
    )
    for lhs, rhs in zip(baseline, isotropic):
        np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))


def test_dark_explicit_isotropic_is_bit_identical_to_accepted_path():
    cat = _runtime_catalog()
    cache = build_observed_density_cache(cat)
    pe = _event(4, np.zeros(4, dtype=np.int32))
    sel = _event(64, np.zeros(64, dtype=np.int32))
    common = dict(
        pop_model="powerlaw+peak",
        max_likelihood_variance=1e6,
        return_diagnostics=True,
    )
    baseline = dark_siren_log_likelihood(
        COSMO, PARAMS, POP, pe, cat, cache, sel, cat, cache,
        1, 4, 128.0, **common,
    )
    isotropic = dark_siren_log_likelihood(
        COSMO, PARAMS, POP, pe, cat, cache, sel, cat, cache,
        1, 4, 128.0,
        angular_model="isotropic",
        angular_params=jnp.asarray([]),
        **common,
    )
    for lhs, rhs in zip(baseline, isotropic):
        np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))


def test_dipole_changes_both_event_and_selection_channels():
    pe = _event(6, np.zeros(6, dtype=np.int32))
    sel = _event(64, np.zeros(64, dtype=np.int32))
    common = dict(
        pop_model="powerlaw+peak",
        max_likelihood_variance=1e6,
        return_diagnostics=True,
    )
    iso = spectral_siren_log_likelihood(
        COSMO, POP, pe, sel, 1, 6, 128.0, **common
    )
    dip = spectral_siren_log_likelihood(
        COSMO,
        POP,
        pe,
        sel,
        1,
        6,
        128.0,
        angular_model="dipole",
        angular_params=jnp.asarray([0.25, -0.10, 0.05]),
        **common,
    )
    assert not np.array_equal(
        np.asarray(iso.event_log_evidence), np.asarray(dip.event_log_evidence)
    )
    assert float(iso.log_mu) != float(dip.log_mu)
