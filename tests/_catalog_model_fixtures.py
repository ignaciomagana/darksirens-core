"""Small in-memory catalog, PE and injection stores shared by the catalog-model tests.

An nside-2 sky (48 rows) with galaxies on most rows, a few empty rows, and
GW samples spread over the sky, so a compact view keeps most but not all
rows. Nothing here is read from disk.
"""

from __future__ import annotations

import numpy as np

from darksirens import Cosmology
from darksirens.catalog.io import CatalogStore
from darksirens.catalog.types import GalaxyCatalog
from darksirens.gw.types import GWStore, SelectionStore
from darksirens.population import get_fixed_population_params, pop_model_prior_parser

FIT = ("m1det", "q", "dL", "chieff")
MODEL = "powerlaw+peak"
_, _, _LABELS, _, _ = pop_model_prior_parser(MODEL)
POPULATION_LABELS = tuple(str(label) for label in _LABELS)
#: Every population parameter fixed at the registry fiducials, with the
#: merger-rate slope gamma pinned explicitly (it is the last entry).
FIXED_POPULATION = {
    label: float(value)
    for label, value in zip(POPULATION_LABELS, get_fixed_population_params(MODEL))
}
FIXED_POPULATION[POPULATION_LABELS[-1]] = 0.0
COSMOLOGY = Cosmology(H0=(50.0, 90.0), Om0=0.3075)
NSIDE = 2
NPIX = 12 * NSIDE * NSIDE


def catalog_store(seed=11, *, z_depth=None, n_max=6, empty=(3, 17, 30), z_hi=0.28, nside=NSIDE):
    """A standardized full-sky catalog store (nside 2 by default) with a few empty rows."""
    rng = np.random.default_rng(seed)
    npix = 12 * nside * nside
    ngals = rng.integers(1, n_max + 1, npix).astype(np.int32)
    ngals[list(empty)] = 0
    zgals = np.full((npix, n_max), 100.0)
    dzgals = np.ones((npix, n_max))
    wgals = np.zeros((npix, n_max))
    for row, n in enumerate(ngals):
        zgals[row, :n] = np.sort(rng.uniform(0.02, z_hi, n))
        dzgals[row, :n] = 0.01 * (1.0 + zgals[row, :n])
        wgals[row, :n] = rng.uniform(0.5, 2.0, n)
    return CatalogStore(
        path=f"catalog-fixture-{seed}.h5",
        nside=nside,
        z_depth=z_depth,
        catalog=GalaxyCatalog(
            apix=np.pi / (3.0 * nside * nside),
            zgals=zgals,
            dzgals=dzgals,
            wgals=wgals,
            ngals=ngals,
            unique_pixels=None,
        ),
    )


def _draw(rng, n, dl_lo, dl_hi, per_event):
    k = n // per_event
    rep = lambda a: np.repeat(a, per_event)  # noqa: E731
    m1 = rep(rng.uniform(25.0, 45.0, k)) * np.exp(rng.normal(0.0, 0.05, n))
    q = np.clip(rep(rng.uniform(0.6, 0.95, k)) + rng.normal(0.0, 0.05, n), 0.1, 1.0)
    ra = np.mod(rep(rng.uniform(0.0, 2.0 * np.pi, k)) + rng.normal(0.0, 0.08, n), 2.0 * np.pi)
    dec = np.clip(
        rep(np.arcsin(rng.uniform(-1.0, 1.0, k))) + rng.normal(0.0, 0.08, n), -1.5, 1.5
    )
    dL = rep(rng.uniform(dl_lo, dl_hi, k)) * np.exp(rng.normal(0.0, 0.12, n))
    return dict(
        m1det=m1, m2det=q * m1, dL=dL, chieff=rng.normal(0.0, 0.05, n), ra=ra, dec=dec
    )


def gw_stores(seed=3, n_events=6, nsamp=8, n_sel=768):
    """In-memory PE and detected-injection stores (chi_eff basis)."""
    rng = np.random.default_rng(seed)
    pe = _draw(rng, n_events * nsamp, 300.0, 1100.0, nsamp)
    sel = _draw(rng, n_sel, 200.0, 1400.0, 1)
    events = GWStore(
        format_version="fixture",
        path="pe-fixture.h5",
        fit_columns=FIT,
        columns=pe,
        attrs={},
        n_events=n_events,
        nsamp=nsamp,
        prior_wt=np.full(n_events * nsamp, 1.0 / nsamp),
        event_names=tuple(f"fixture{i}" for i in range(n_events)),
    )
    injections = SelectionStore(
        format_version="fixture",
        path="selection-fixture.h5",
        fit_columns=FIT,
        columns=sel,
        attrs={},
        n_injections=n_sel,
        ndraw=2.0 * n_sel,
        prior_wt=np.ones(n_sel),
    )
    return events, injections


def theta_of(analysis, values=None, *, seed=None):
    """A coordinate vector: ``values`` by label, the rest at the prior midpoints
    (or uniform draws inside the bounds when ``seed`` is given)."""
    plan = analysis.parameters
    rng = None if seed is None else np.random.default_rng(seed)
    out = []
    for label, lo, hi in zip(plan.labels, plan.lower, plan.upper):
        if values and label in values:
            out.append(float(values[label]))
        elif rng is not None:
            out.append(float(rng.uniform(lo, hi)))
        else:
            out.append(0.5 * (float(lo) + float(hi)))
    return np.asarray(out)
