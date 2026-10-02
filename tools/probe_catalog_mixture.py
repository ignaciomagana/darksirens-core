#!/usr/bin/env python3
"""Fixed-theta parity probe for the field-weighted (multi-catalog) mixture.

Run in separate Python processes with either the pinned legacy repository or
the reconstructed ``src`` directory first on ``PYTHONPATH``: both expose the
package name ``darksirens`` and must never share an interpreter.

The candidate arm goes through the public surface: ``ds.model(catalog=[A, B],
catalog_sky_weighting="field", ...)`` and ``bind_analysis`` on in-memory
stores. The legacy arm calls the frozen ``darksiren_log_likelihood`` with
``catalog_sky_weighting="field"``, ``n_catalogs=K``, ``mixture_surveys``,
``mixture_em_catalogs_*`` and ``mixture_log_weights`` from its own
``_sticks_to_log_weights``, on full-sky catalogs carrying the survey-global
field inputs its loader builds (``build_field_normalization_inputs``,
``build_field_depth_inputs``) and, for a coverage fraction, ``f_p_rows`` /
``field_f_p_occ`` / ``field_f_p_empty_sum``. The two catalogs have different
HEALPix resolutions (nside 2 and 4), so each sample has a row in each.

The population is one explicit vector in both arms with the merger-rate slope
gamma pinned (``GAMMA``).

Legacy stores the count-ratio normaliser's observed density in float32
(``field_dN_obs_s``); core keeps it in float64. ``--legacy-f64-obs`` (legacy
arm) replaces legacy's table by the float64 one, which isolates that storage
difference: the comparison of the count-ratio cells is at rounding level
with it and ~1e-8 relative without it.

``--compare LEGACY CANDIDATE --rtol R`` checks two outputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REFERENCE_SHA = "c042527238bd71421b792936bc48c3b815b90d6d"
POP_MODEL = "powerlaw+peak"
GAMMA = 0.0
N_EVENTS = 4
NSAMP = 8
N_SEL = 256
N_DRAW = 2.0 * N_SEL
MAX_VARIANCE = 1.0e3
OM0 = 0.3075
SCHECHTER = (
    dict(m_lim=19.0, Mstar_hat=-20.6, alpha=-1.05, M_faint_offset=5.0),
    dict(m_lim=17.5, Mstar_hat=-21.0, alpha=-0.8, M_faint_offset=5.0),
)

#: (name, K, completeness, z_depth, row_fraction, fixed kernel params, points).
#: A point: per catalog (log10n0, delta, sigma_kde[, Mstar_hat, alpha]), then
#: (H0, fcat_2, ...). With fixed kernel params (delta, sigma_kde) the
#: candidate pins each catalog's kernel and the legacy arm does not.
CELLS = (
    ("k1_count", 1, "incomplete", None, False, None,
     (((-2.0, 0.0, 0.0),), (67.74,)),
     (((-2.4, 0.5, 0.004),), (58.0,))),
    ("k1_selection_depth", 1, "selection", 0.22, False, None,
     (((-2.0, 0.0, 0.0, -20.6, -1.05),), (67.74,)),
     (((-2.3, 0.4, 0.003, -20.9, -1.3),), (74.0,))),
    ("k2_count", 2, "incomplete", None, False, None,
     (((-2.0, 0.0, 0.0), (-2.5, 0.0, 0.0)), (67.74, 0.3)),
     (((-2.4, 0.5, 0.004), (-1.8, -0.3, 0.0)), (58.0, 0.8)),
     (((-1.7, -0.4, 0.01), (-2.9, 0.9, 0.006)), (81.0, 0.05))),
    ("k2_count_depth", 2, "incomplete", 0.22, False, None,
     (((-2.0, 0.0, 0.0), (-2.5, 0.0, 0.0)), (67.74, 0.3)),
     (((-2.4, 0.5, 0.004), (-1.8, -0.3, 0.002)), (62.0, 0.6))),
    ("k2_count_depth_pinned", 2, "incomplete", 0.22, False, (0.4, 0.003),
     (((-2.0,), (-2.5,)), (67.74, 0.3)),
     (((-2.4,), (-1.8,)), (55.0, 0.7))),
    ("k2_selection", 2, "selection", None, False, None,
     (((-2.0, 0.0, 0.0, -20.6, -1.05), (-2.5, 0.0, 0.0, -21.0, -0.8)), (67.74, 0.3)),
     (((-2.3, 0.4, 0.005, -20.9, -1.3), (-1.9, 0.2, 0.0, -20.7, -1.1)), (60.0, 0.75))),
    ("k2_selection_depth_fp", 2, "selection", 0.22, True, None,
     (((-2.0, 0.0, 0.0, -20.6, -1.05), (-2.5, 0.0, 0.0, -21.0, -0.8)), (67.74, 0.3)),
     (((-2.2, 0.6, 0.002, -20.2, -0.9), (-2.8, -0.5, 0.004, -21.3, -1.2)), (77.0, 0.45))),
    ("k3_count", 3, "incomplete", None, False, None,
     (((-2.0, 0.0, 0.0), (-2.5, 0.0, 0.0), (-2.2, 0.3, 0.0)), (67.74, 0.3, 0.4))),
)


def _cell_points(cell):
    return cell[6:]


def _catalog(seed, nside, z_depth):
    rng = np.random.default_rng(seed)
    npix = 12 * nside * nside
    n_max = 5
    ngals = rng.integers(1, n_max + 1, npix).astype(np.int32)
    ngals[rng.choice(npix, size=max(2, npix // 12), replace=False)] = 0
    zgals = np.full((npix, n_max), 100.0)
    dzgals = np.ones((npix, n_max))
    wgals = np.zeros((npix, n_max))
    z_hi = 0.28 if z_depth is None else 0.3
    for row, n in enumerate(ngals):
        zgals[row, :n] = np.sort(rng.uniform(0.02, z_hi, n))
        dzgals[row, :n] = 0.01 * (1.0 + zgals[row, :n])
        wgals[row, :n] = rng.uniform(0.5, 2.0, n)
    return dict(zgals=zgals, dzgals=dzgals, wgals=wgals, ngals=ngals,
                apix=np.pi / (3.0 * nside * nside), nside=nside)


def _row_fraction(seed, npix):
    # Multiples of 1/4: exact in float32, legacy's f_p storage.
    rng = np.random.default_rng(seed)
    return rng.integers(0, 5, npix) / 4.0


def _samples():
    rng = np.random.default_rng(20261002)

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
    return pe, sel


def _cell_catalogs(K, z_depth):
    nsides = (2, 4, 2)[:K]
    return [_catalog(100 + k, nsides[k], z_depth) for k in range(K)]


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
    from darksirens.selection.catalog import SchechterMagnitudeSelection

    print("darksirens:", ds.__file__)
    pe, sel = _samples()
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
    for cell in CELLS:
        name, K, completeness, z_depth, with_fp, fixed_kernel = cell[:6]
        cats = _cell_catalogs(K, z_depth)
        stores = [
            CatalogStore(
                path=f"probe-catalog-{k}", nside=c["nside"], z_depth=z_depth,
                catalog=GalaxyCatalog(
                    apix=c["apix"], zgals=c["zgals"], dzgals=c["dzgals"],
                    wgals=c["wgals"], ngals=c["ngals"], unique_pixels=None,
                ),
            )
            for k, c in enumerate(cats)
        ]
        kwargs = {}
        suffixes = [""] + [f"_c{k + 1}" for k in range(1, K)]
        if completeness == "selection":
            kwargs["selection"] = [SchechterMagnitudeSelection(**SCHECHTER[k]) for k in range(K)]
            kwargs["survey_priors"] = {
                label: bounds
                for s in suffixes
                for label, bounds in ((f"Mstar_hat{s}", (-23.0, -18.0)), (f"alpha{s}", (-1.9, 0.0)))
            }
            if with_fp:
                kwargs["row_fraction"] = [
                    _row_fraction(200 + k, 12 * cats[k]["nside"] ** 2) for k in range(K)
                ]
        if fixed_kernel is not None:
            kwargs["fixed_survey"] = {
                f"{p}{s}": v for s in suffixes for p, v in zip(("delta", "sigma_kde"), fixed_kernel)
            }
        analysis = ds.model(
            cosmology=ds.Cosmology(H0=(20.0, 140.0), Om0=OM0),
            population=ds.Population(POP_MODEL, fixed=dict(zip(labels, population))),
            catalog=stores,
            catalog_sky_weighting="field",
            completeness=completeness,
            **kwargs,
        )
        if fixed_kernel is not None:
            assert analysis.parameters.kernel_pin_active
        bound = bind_analysis(analysis, events=events, injections=injections,
                              max_likelihood_variance=MAX_VARIANCE)
        values = []
        for per_catalog, (H0, *sticks) in _cell_points(cell):
            theta = [H0]
            for k in range(K):
                survey = list(per_catalog[k])
                theta += survey[:1] if fixed_kernel is not None else survey
            theta += list(sticks)
            values.append(float(bound(np.asarray(theta, dtype=np.float64))))
        rows[name] = values
    return rows, population


def _legacy(f64_obs):
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
    from darksirens.inference.parameters import _sticks_to_log_weights
    from darksirens.likelihood.core import darksiren_log_likelihood
    from darksirens.likelihood.events import make_gw_event
    from darksirens.redshift import zgrid
    from darksirens.redshift.completion import (
        _kde_rows,
        build_field_depth_inputs,
        build_field_normalization_inputs,
    )

    print("darksirens:", darksirens.__file__)
    pe, sel = _samples()
    population = _population(get_fixed_population_params)
    rows = {}
    for cell in CELLS:
        name, K, completeness, z_depth, with_fp, fixed_kernel = cell[:6]
        cats = _cell_catalogs(K, z_depth)

        def event(cols):
            pix = [
                np.asarray(hp.ang2pix(c["nside"], np.pi / 2.0 - cols["dec"], cols["ra"]), dtype=np.int32)
                for c in cats
            ]
            pixels = pix[0] if K == 1 else np.stack(pix, axis=1)
            return make_gw_event(
                m1det=cols["m1det"], m2det=cols["m2det"], dL=cols["dL"], chieff=cols["chieff"],
                prior_wt=cols["prior_wt"], pixels=pixels,
            )

        gw_pe, gw_sel = event(pe), event(sel)
        em = []
        for k, c in enumerate(cats):
            field = build_field_normalization_inputs(c["zgals"], c["wgals"], c["ngals"])
            obs = field.dN_obs_s
            if f64_obs:
                obs = jnp.asarray(_kde_rows(
                    field.occupied_pixels, jnp.asarray(c["zgals"]), jnp.asarray(c["wgals"]),
                    jnp.asarray(c["ngals"]), 4096,
                ), dtype=jnp.float64)
            extra = dict(
                field_dN_obs_s=obs, field_n_empty=field.n_empty,
                field_N_obs_total=field.N_obs_total,
                field_occupied_pixels=field.occupied_pixels,
            )
            if z_depth is not None:
                depth = build_field_depth_inputs(c["zgals"], c["dzgals"], c["wgals"], c["ngals"])
                extra.update(field_depth_z=depth.z, field_depth_dz=depth.dz, field_depth_c=depth.c)
            if with_fp:
                f = _row_fraction(200 + k, 12 * c["nside"] ** 2)
                occ = np.asarray(field.occupied_pixels)
                empty = np.setdiff1d(np.arange(f.size), occ)
                extra.update(
                    f_p_rows=jnp.asarray(f, dtype=jnp.float32),
                    field_f_p_occ=jnp.asarray(f[occ], dtype=jnp.float32),
                    field_f_p_empty_sum=float(np.sum(f[empty])),
                )
            em.append(EMCatalog(
                apix=c["apix"], zgals=jnp.asarray(c["zgals"]), dzgals=jnp.asarray(c["dzgals"]),
                wgals=jnp.asarray(c["wgals"]), ngals=jnp.asarray(c["ngals"]),
                delta_g_pix_z=jnp.zeros((1, int(zgrid.shape[0]))), dN_obs_kde=None,
                pixel_to_cache_idx=None, unique_pixels=None, **extra,
            ))
        values = []
        for per_catalog, (H0, *sticks) in _cell_points(cell):
            surveys = []
            for k in range(K):
                entries = list(per_catalog[k])
                if fixed_kernel is not None:
                    entries = entries[:1] + list(fixed_kernel)
                log10n0, delta, sigma_kde = entries[:3]
                extra = {}
                if completeness == "selection":
                    fields = dict(SCHECHTER[k])
                    fields["Mstar_hat"], fields["alpha"] = entries[3], entries[4]
                    extra = dict(c_mode=C_MODE_SELECTION_STRUCT,
                                 selection_family=SELECTION_FAMILY_SCHECHTER_STRUCT, **fields)
                surveys.append(SurveyParams(
                    n0=10.0 ** log10n0, z50=1.0, w=0.5, delta=delta, b_miss=1.0,
                    alpha_miss=1.0, sigma_kde=sigma_kde, z_depth=z_depth, **extra,
                ))
            cosmo = CosmoParams(H0=H0, Om0=OM0, w0=-1.0, wa=0.0)
            mix = {}
            if K >= 2:
                mix = dict(
                    n_catalogs=K,
                    mixture_surveys=tuple(surveys[1:]),
                    mixture_em_catalogs_pe=tuple(em[1:]),
                    mixture_em_catalogs_sel=tuple(em[1:]),
                    mixture_log_weights=_sticks_to_log_weights(jnp.asarray(sticks)),
                )
            values.append(float(darksiren_log_likelihood(
                cosmo, surveys[0], jnp.asarray(population), gw_pe, em[0], gw_sel, em[0],
                N_EVENTS, NSAMP, N_DRAW, POP_MODEL, "dark_sirens",
                max_likelihood_variance=MAX_VARIANCE,
                catalog_sky_weighting="field",
                **mix,
            )))
        rows[name] = values
    return rows, population


def _compare(legacy_path, candidate_path, rtol, only=None):
    legacy = json.loads(Path(legacy_path).read_text())
    candidate = json.loads(Path(candidate_path).read_text())
    assert legacy["population"] == candidate["population"], "population vectors differ"
    worst = 0.0
    for name, values in legacy["results"].items():
        if only and not any(name.startswith(prefix) for prefix in only):
            continue
        got = candidate["results"][name]
        for a, b in zip(values, got):
            if not (math.isfinite(a) and math.isfinite(b)):
                raise AssertionError(f"{name}: non-finite {a} vs {b}")
            diff = abs(a - b)
            worst = max(worst, diff)
            if diff > rtol * max(1.0, abs(a)):
                raise AssertionError(f"{name}: legacy={a!r} candidate={b!r} |d|={diff:.3e}")
        print(f"{name:24s} " + " ".join(f"{abs(a - b):.2e}" for a, b in zip(values, got)))
    print(f"max |dlogL| = {worst:.3e} (rtol {rtol:.1e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"))
    parser.add_argument("--out")
    parser.add_argument("--legacy-f64-obs", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("LEGACY", "CANDIDATE"))
    parser.add_argument("--only", nargs="*", default=None,
                        help="compare only the cells whose names start with these")
    parser.add_argument("--rtol", type=float, default=1e-12)
    args = parser.parse_args()
    if args.compare:
        _compare(*args.compare, args.rtol, args.only)
        return
    if args.implementation == "legacy":
        rows, population = _legacy(args.legacy_f64_obs)
    else:
        rows, population = _candidate()
    payload = {
        "implementation": args.implementation,
        "reference_sha": REFERENCE_SHA,
        "pop_model": POP_MODEL,
        "gamma": GAMMA,
        "legacy_f64_obs": bool(args.legacy_f64_obs),
        "population": [float(x) for x in population],
        "results": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
