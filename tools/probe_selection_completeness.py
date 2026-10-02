#!/usr/bin/env python3
"""Fixed-theta parity probe for ``ds.model(..., completeness="selection")``.

Run in separate Python processes with either the pinned legacy repository or
the reconstructed ``src`` directory first on ``PYTHONPATH``: both expose the
package name ``darksirens`` and must never share an interpreter.

The candidate arm goes through the public surface (``ds.model`` with
``completeness="selection"``, ``bind_analysis`` on in-memory stores); the
legacy arm calls the frozen ``darksiren_log_likelihood`` with
``c_mode="selection"`` (``SurveyParams.c_mode = C_MODE_SELECTION_STRUCT``) on
the full-sky catalog, the conditional sky weighting, and the same samples.
The population is fixed at one explicit vector in both arms, its merger-rate
slope gamma pinned (``GAMMA``), so the two registries' fiducials never enter.

``--compare LEGACY CANDIDATE --rtol R`` checks the two outputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REFERENCE_SHA = "c042527238bd71421b792936bc48c3b815b90d6d"
POP_MODEL = "powerlaw+peak"
#: Explicit merger-rate slope of the fixed population (the last entry).
GAMMA = 0.0
NSIDE = 2
NPIX = 12 * NSIDE * NSIDE
N_EVENTS = 4
NSAMP = 8
N_SEL = 256
N_DRAW = 2.0 * N_SEL
MAX_VARIANCE = 1.0e3
OM0 = 0.3075

#: (name, family, z_depth, selection fields, points). A point is
#: (H0, log10n0, delta, sigma_kde, nuisance...).
CELLS = (
    (
        "gaussian",
        "gaussian",
        None,
        dict(m_lim=21.0, M0hat=-20.3, sigma_M=0.72, k_corr_coeffs=(1.13, -4.89, 8.59)),
        ((67.74, -2.0, 0.0, 0.0, -20.3, 0.72), (58.0, -2.4, 0.6, 0.004, -20.6, 0.9),
         (81.0, -1.7, -0.4, 0.01, -19.9, 0.55)),
    ),
    (
        "gaussian_depth",
        "gaussian",
        0.22,
        dict(m_lim=21.0, M0hat=-20.3, sigma_M=0.72, k_corr_coeffs=()),
        ((67.74, -2.0, 0.0, 0.0, -20.3, 0.72), (74.0, -2.2, 0.9, 0.003, -20.1, 0.8)),
    ),
    (
        "schechter",
        "schechter",
        None,
        dict(m_lim=19.0, Mstar_hat=-20.6, alpha=-1.05, M_faint_offset=5.0),
        ((67.74, -2.0, 0.0, 0.0, -20.6, -1.05), (60.0, -2.3, 0.4, 0.005, -20.9, -1.3),
         (85.0, -1.8, -0.2, 0.0, -20.2, -0.7)),
    ),
    (
        "schechter_depth",
        "schechter",
        0.22,
        dict(m_lim=20.0, Mstar_hat=-19.63, alpha=-1.07, M_faint_offset=-0.094),
        ((67.74, -2.0, 0.0, 0.0, -19.63, -1.07), (70.0, -2.6, 0.3, 0.002, -19.8, -1.2)),
    ),
)
NUISANCES = {"gaussian": ("M0hat", "sigma_M"), "schechter": ("Mstar_hat", "alpha")}


def _fixture():
    """The shared catalog and GW samples (numpy only; identical in both arms)."""
    rng = np.random.default_rng(20261002)
    n_max = 5
    ngals = rng.integers(1, n_max + 1, NPIX).astype(np.int32)
    ngals[[3, 17, 30]] = 0
    zgals = np.full((NPIX, n_max), 100.0)
    dzgals = np.ones((NPIX, n_max))
    wgals = np.zeros((NPIX, n_max))
    for row, n in enumerate(ngals):
        zgals[row, :n] = np.sort(rng.uniform(0.02, 0.28, n))
        dzgals[row, :n] = 0.01 * (1.0 + zgals[row, :n])
        wgals[row, :n] = rng.uniform(0.5, 2.0, n)

    def draw(n, dl_lo, dl_hi, per_event):
        k = n // per_event
        rep = lambda a: np.repeat(a, per_event)  # noqa: E731
        m1 = rep(rng.uniform(25.0, 45.0, k)) * np.exp(rng.normal(0.0, 0.05, n))
        q = np.clip(rep(rng.uniform(0.6, 0.95, k)) + rng.normal(0.0, 0.05, n), 0.1, 1.0)
        ra = np.mod(rep(rng.uniform(0.0, 2.0 * np.pi, k)) + rng.normal(0.0, 0.08, n), 2.0 * np.pi)
        dec = np.clip(rep(np.arcsin(rng.uniform(-1.0, 1.0, k))) + rng.normal(0.0, 0.08, n), -1.5, 1.5)
        dL = rep(rng.uniform(dl_lo, dl_hi, k)) * np.exp(rng.normal(0.0, 0.12, n))
        return dict(m1det=m1, m2det=q * m1, dL=dL, chieff=rng.normal(0.0, 0.05, n), ra=ra, dec=dec)

    pe = draw(N_EVENTS * NSAMP, 300.0, 1100.0, NSAMP)
    pe["prior_wt"] = np.full(N_EVENTS * NSAMP, 1.0 / NSAMP)
    sel = draw(N_SEL, 200.0, 1400.0, 1)
    sel["prior_wt"] = np.ones(N_SEL)
    catalog = dict(zgals=zgals, dzgals=dzgals, wgals=wgals, ngals=ngals, apix=np.pi / (3.0 * NSIDE**2))
    return catalog, pe, sel


def _population(get_fixed_population_params):
    theta = np.asarray(get_fixed_population_params(POP_MODEL), dtype=np.float64).copy()
    theta[-1] = GAMMA
    return theta


def _candidate():
    import jax

    jax.config.update("jax_enable_x64", True)

    import darksirens as ds
    from darksirens.catalog.io import CatalogStore
    from darksirens.catalog.types import GalaxyCatalog
    from darksirens.gw.types import GWStore, SelectionStore
    from darksirens.population import get_fixed_population_params, pop_model_prior_parser
    from darksirens.runtime_binding import bind_analysis
    from darksirens.selection.catalog import GaussianMagnitudeSelection, SchechterMagnitudeSelection

    print("darksirens:", ds.__file__)
    cat, pe, sel = _fixture()
    fit = ("m1det", "q", "dL", "chieff")
    events = GWStore(
        format_version="probe", path="probe-pe", fit_columns=fit,
        columns={k: v for k, v in pe.items() if k != "prior_wt"}, attrs={},
        n_events=N_EVENTS, nsamp=NSAMP, prior_wt=pe["prior_wt"],
        event_names=tuple(f"e{i}" for i in range(N_EVENTS)),
    )
    injections = SelectionStore(
        format_version="probe", path="probe-sel", fit_columns=fit,
        columns={k: v for k, v in sel.items() if k != "prior_wt"}, attrs={},
        n_injections=N_SEL, ndraw=N_DRAW, prior_wt=sel["prior_wt"],
    )
    population = _population(get_fixed_population_params)
    labels = [str(x) for x in pop_model_prior_parser(POP_MODEL)[2]]
    rows = {}
    for name, family, z_depth, fields, points in CELLS:
        store = CatalogStore(
            path="probe-catalog", nside=NSIDE, z_depth=z_depth,
            catalog=GalaxyCatalog(
                apix=cat["apix"], zgals=cat["zgals"], dzgals=cat["dzgals"],
                wgals=cat["wgals"], ngals=cat["ngals"], unique_pixels=None,
            ),
        )
        model_cls = GaussianMagnitudeSelection if family == "gaussian" else SchechterMagnitudeSelection
        names = NUISANCES[family]
        analysis = ds.model(
            cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=OM0),
            population=ds.Population(POP_MODEL, fixed=dict(zip(labels, population))),
            catalog=store,
            completeness="selection",
            selection=model_cls(**fields),
            survey_priors={n: ((-23.0, -18.0) if n.startswith("M") else
                               ((0.05, 3.0) if n == "sigma_M" else (-1.9, 0.0)))
                           for n in names},
        )
        assert analysis.parameters.labels == ("H0", "log10n0", "delta", "sigma_kde") + names
        bound = bind_analysis(analysis, events=events, injections=injections,
                              max_likelihood_variance=MAX_VARIANCE)
        rows[name] = [float(bound(np.asarray(point, dtype=np.float64))) for point in points]
    return rows, population


def _legacy():
    import healpy as hp
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    import darksirens
    from darksirens.core.types import (
        C_MODE_SELECTION_STRUCT,
        SELECTION_FAMILY_SCHECHTER_STRUCT,
        CosmoParams,
        EMCatalog,
        SurveyParams,
    )
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.core import darksiren_log_likelihood
    from darksirens.likelihood.events import make_gw_event
    from darksirens.redshift import zgrid

    print("darksirens:", darksirens.__file__)
    cat, pe, sel = _fixture()
    population = _population(get_fixed_population_params)

    def event(cols):
        pixels = hp.ang2pix(NSIDE, np.pi / 2.0 - cols["dec"], cols["ra"])
        return make_gw_event(
            m1det=cols["m1det"], m2det=cols["m2det"], dL=cols["dL"], chieff=cols["chieff"],
            prior_wt=cols["prior_wt"], pixels=np.asarray(pixels, dtype=np.int32),
        )

    gw_pe, gw_sel = event(pe), event(sel)
    catalog = EMCatalog(
        apix=cat["apix"], zgals=jnp.asarray(cat["zgals"]), dzgals=jnp.asarray(cat["dzgals"]),
        wgals=jnp.asarray(cat["wgals"]), ngals=jnp.asarray(cat["ngals"]),
        delta_g_pix_z=jnp.zeros((1, int(zgrid.shape[0]))), dN_obs_kde=None,
        pixel_to_cache_idx=None, unique_pixels=None,
    )
    rows = {}
    for name, family, z_depth, fields, points in CELLS:
        values = []
        for H0, log10n0, delta, sigma_kde, *nuisance in points:
            extra = dict(fields)
            extra.update(zip(NUISANCES[family], nuisance))
            if family == "gaussian":
                kz = tuple(extra.pop("k_corr_coeffs") or ()) or None
                extra["k_corr_coeffs"] = kz
            else:
                extra["selection_family"] = SELECTION_FAMILY_SCHECHTER_STRUCT
            survey = SurveyParams(
                n0=10.0 ** log10n0, z50=1.0, w=0.5, delta=delta, b_miss=1.0,
                alpha_miss=1.0, sigma_kde=sigma_kde, z_depth=z_depth,
                c_mode=C_MODE_SELECTION_STRUCT, **extra,
            )
            cosmo = CosmoParams(H0=H0, Om0=OM0, w0=-1.0, wa=0.0)
            values.append(float(darksiren_log_likelihood(
                cosmo, survey, jnp.asarray(population), gw_pe, catalog, gw_sel, catalog,
                N_EVENTS, NSAMP, N_DRAW, POP_MODEL, "dark_sirens",
                max_likelihood_variance=MAX_VARIANCE,
                catalog_sky_weighting="conditional",
            )))
        rows[name] = values
    return rows, population


def _compare(legacy_path, candidate_path, rtol):
    legacy = json.loads(Path(legacy_path).read_text())
    candidate = json.loads(Path(candidate_path).read_text())
    assert legacy["population"] == candidate["population"], "population vectors differ"
    worst = 0.0
    for name, values in legacy["results"].items():
        got = candidate["results"][name]
        for a, b in zip(values, got):
            if not (math.isfinite(a) and math.isfinite(b)):
                raise AssertionError(f"{name}: non-finite {a} vs {b}")
            diff = abs(a - b)
            worst = max(worst, diff)
            if diff > rtol * max(1.0, abs(a)):
                raise AssertionError(f"{name}: legacy={a!r} candidate={b!r} |d|={diff:.3e}")
        print(f"{name:18s} " + " ".join(f"{abs(a - b):.2e}" for a, b in zip(values, got)))
    print(f"max |dlogL| = {worst:.3e} (rtol {rtol:.1e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"))
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2, metavar=("LEGACY", "CANDIDATE"))
    parser.add_argument("--rtol", type=float, default=1e-12)
    args = parser.parse_args()
    if args.compare:
        _compare(*args.compare, args.rtol)
        return
    rows, population = _legacy() if args.implementation == "legacy" else _candidate()
    payload = {
        "implementation": args.implementation,
        "reference_sha": REFERENCE_SHA,
        "pop_model": POP_MODEL,
        "gamma": GAMMA,
        "population": [float(x) for x in population],
        "results": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
