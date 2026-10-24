"""The uninformative-catalog null of the count-ratio completeness.

Galaxies drawn Poisson in comoving volume with a known completeness and GW
events drawn independently of them carry no host information, so with the
catalog's own ``n0`` the host density is the expected density and
``ln L_cat(H0) - ln L_spec(H0)`` is flat up to shot noise. These tests pin
that null at fixed ``n0`` (``h_scaled`` at the truth) over H0 = 25 to 100:

- spectroscopic redshifts under the default (pooled) count ratio: flat;
- photometric redshifts (0.03 (1 + z)) under the default: not flat, because
  galaxies scattered toward z = 0 push the pooled ratio of point redshifts
  above 1 there and the clip removes the missing hosts. That is a known defect
  of the default (it shipped in #68 with the suite green). Its test asserts
  the null and is an expected failure: it records the measured spread
  (``default_null_spread_nats`` in the test report) and passes, without
  failing the suite, the day the default is fixed;
- photometric redshifts under ``count_ratio="pooled_kernel"``: flat.

Shot noise of the difference over four pairs of catalog and GW seeds: a
spread (max - min over H0) of 0.26 to 0.90 nats for the flat cases, against
5.3 to 8.0 for the photometric default. The seeds here are fixed. The
catalog is nside 2 with about 1.8e5 galaxies; 24 events of 64 samples and
2000 injections.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import darksirens as ds
from darksirens import Population
from darksirens.catalog.io import CatalogStore
from darksirens.catalog.types import GalaxyCatalog
from darksirens.cosmology.distances import dV_of_z
from darksirens.runtime_binding import bind_analysis

from _catalog_model_fixtures import FIXED_POPULATION, MODEL, gw_stores

jax.config.update("jax_enable_x64", True)

NSIDE = 2
NPIX = 12 * NSIDE**2
DEPTH = 0.25
H0_TRUE = 70.0
OM0 = 0.3075
LOG10N0 = -3.6  # h^3 Mpc^-3 (n0_units="h_scaled")
H0S = (25.0, 30.0, 40.0, 55.0, 70.0, 85.0, 100.0)
#: Spread of ln L_cat - ln L_spec over H0S below which the null holds.
FLAT = 1.5
POPULATION = Population(MODEL, fixed=FIXED_POPULATION)
COSMOLOGY = ds.Cosmology(H0=(20.0, 140.0), Om0=OM0)
EVENTS, INJECTIONS = gw_stores(seed=3, n_events=24, nsamp=64, n_sel=2000)


def _completeness(kind, z):
    if kind == "constant":
        return np.where(z <= DEPTH, 0.5, 0.0)
    # Magnitude-limit-like roll-off: 1 at z -> 0, 0.5 at z = 0.12.
    return np.where(z <= DEPTH, 1.0 / (1.0 + np.exp((z - 0.12) / 0.03)), 0.0)


def _store(sigma, kind="constant", seed=1):
    """A full-sky nside-2 store of galaxies Poisson in comoving volume at the
    true H0, thinned by the completeness, with redshift errors ``sigma (1+z)``
    (``dzgals`` 1e-4 without)."""
    rng = np.random.default_rng(seed)
    n0 = 10.0**LOG10N0 * (H0_TRUE / 100.0) ** 3
    zf = np.linspace(0.0, DEPTH, 20_001)
    dV = np.asarray(dV_of_z(jnp.asarray(zf), H0_TRUE, OM0, -1.0, 0.0))
    density = _completeness(kind, zf) * n0 * 4.0 * np.pi * dV
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(zf))])
    n = rng.poisson(cdf[-1])
    zt = np.interp(rng.uniform(0.0, cdf[-1], n), cdf, zf)
    z = zt + sigma * (1.0 + zt) * rng.normal(size=n) if sigma else zt
    dz = sigma * (1.0 + np.maximum(z, 0.0)) if sigma else np.full(n, 1.0e-4)
    pixel = rng.integers(0, NPIX, n)
    ngals = np.bincount(pixel, minlength=NPIX)
    zgals = np.full((NPIX, ngals.max()), 100.0)
    dzgals = np.ones(zgals.shape)
    wgals = np.zeros(zgals.shape)
    for row in range(NPIX):
        mine = pixel == row
        zgals[row, : ngals[row]] = z[mine]
        dzgals[row, : ngals[row]] = dz[mine]
        wgals[row, : ngals[row]] = 1.0
    catalog = GalaxyCatalog(
        apix=np.pi / (3.0 * NSIDE**2), zgals=zgals, dzgals=dzgals, wgals=wgals,
        ngals=ngals.astype(np.int32), unique_pixels=None,
    )
    return CatalogStore(path=f"null-{kind}-{sigma}-{seed}.h5", nside=NSIDE, z_depth=DEPTH,
                        catalog=catalog)


def _bind(analysis):
    return bind_analysis(
        analysis, events=EVENTS, injections=INJECTIONS, max_likelihood_variance=1.0e9
    )


@pytest.fixture(scope="module")
def spectral():
    bound = _bind(ds.model(cosmology=COSMOLOGY, population=POPULATION))
    return np.asarray([float(bound(np.asarray([h]))) for h in H0S])


def _null(spectral, store, **kwargs):
    """``ln L_cat - ln L_spec`` over H0S, ``n0`` fixed at the truth."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analysis = ds.model(
            cosmology=COSMOLOGY, population=POPULATION, catalog=store, n0_units="h_scaled",
            fixed_survey={"log10n0": LOG10N0, "delta": 0.0, "sigma_kde": 1.0e-3},
            allow_out_of_prior=True, **kwargs,
        )
        bound = _bind(analysis)
    assert analysis.parameters.labels == ("H0",)
    return np.asarray([float(bound(np.asarray([h]))) for h in H0S]) - spectral


def test_the_null_holds_with_spectroscopic_redshifts_by_default(spectral):
    diff = _null(spectral, _store(0.0))
    assert np.ptp(diff) < FLAT, diff


@pytest.fixture(scope="module")
def photometric():
    return _store(0.03)


def test_the_photometric_null_holds_with_kernel_counts_conditional(
    spectral, photometric
):
    kernel = _null(spectral, photometric, count_ratio="pooled_kernel")
    assert np.ptp(kernel) < FLAT, kernel


@pytest.mark.xfail(
    reason=(
        "known defect of the default pooled count ratio with photometric redshifts: "
        "galaxies scattered toward z = 0 push the pooled ratio of point redshifts "
        "above 1 and the clip removes the missing hosts (spread 5.3 to 8.0 nats over "
        "four seed pairs); count_ratio='pooled_kernel' holds the null"
    ),
    strict=False,
)
def test_the_photometric_null_holds_by_default(spectral, photometric, record_property):
    default = _null(spectral, photometric)
    record_property("default_null_spread_nats", float(np.ptp(default)))
    assert np.ptp(default) < FLAT, default


@pytest.mark.slow
@pytest.mark.parametrize(
    "kind, weighting", [("falling", "conditional"), ("constant", "field")]
)
def test_the_photometric_null_holds_with_kernel_counts(spectral, kind, weighting):
    diff = _null(
        spectral, _store(0.03, kind), count_ratio="pooled_kernel",
        catalog_sky_weighting=weighting,
    )
    assert np.ptp(diff) < FLAT, diff
