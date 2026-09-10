#!/usr/bin/env python3
"""Phase 5C fixed-theta parity probe for ordinary/bright siren likelihoods.

Run in separate Python processes with either the pinned legacy repository or the
reconstructed ``src`` directory first on ``PYTHONPATH``. Both implementations
expose the package name ``darksirens`` and must never share an interpreter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REFERENCE_SHA = "c042527238bd71421b792936bc48c3b815b90d6d"
POP_MODEL = "powerlaw+peak"
N_EVENTS = 1
NSAMP = 2
N_SEL = 8
N_DRAW = float(N_SEL)
MAX_VARIANCE = 1.0
POINT_FRACTIONS = (0.50, 0.35, 0.65)
POINT_NAMES = ("f050", "f035", "f065")
CELLS = ("plain_full", "plain_compact", "complete_volume", "complete_zero", "bright")
NPIX = 12
APIX = float(np.pi / 3.0)


def _population_points(pop_model_prior_parser, get_fixed_population_params):
    lower, upper, labels, *_ = pop_model_prior_parser(POP_MODEL)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    fid = np.asarray(get_fixed_population_params(POP_MODEL), dtype=np.float64)
    if not labels:
        raise RuntimeError("population parser returned no labels")
    points = []
    for name, frac in zip(POINT_NAMES, POINT_FRACTIONS):
        theta = fid.copy()
        theta[0] = lower[0] + frac * (upper[0] - lower[0])
        points.append((name, theta))
    return points


def _base_arrays(*, compact=False, sparse=False):
    nrows = 1 if compact else NPIX
    zgals = np.full((nrows, 1), 0.10, dtype=np.float64)
    dzgals = np.full((nrows, 1), 0.02, dtype=np.float64)
    wgals = np.ones((nrows, 1), dtype=np.float64)
    ngals = np.ones(nrows, dtype=np.int32)
    if sparse:
        if compact:
            raise ValueError("sparse fixture is only defined for the full catalog")
        zgals[2, 0] = 0.0
        wgals[2, 0] = 0.0
        ngals[2] = 0

    pe_pixels = np.zeros(NSAMP, dtype=np.int32) if compact else np.array([7, 7], dtype=np.int32)
    sel_pixels = (
        np.zeros(N_SEL, dtype=np.int32)
        if compact
        else np.array([2, 7, 2, 7, 2, 7, 2, 7], dtype=np.int32)
    )

    return {
        "zgals": zgals,
        "dzgals": dzgals,
        "wgals": wgals,
        "ngals": ngals,
        "unique_pixels": np.array([0], dtype=np.int32) if compact else None,
        "pe": {
            "m1det": np.array([36.0, 38.0]),
            "m2det": np.array([28.8, 30.4]),
            "dL": np.array([460.0, 500.0]),
            "chieff": np.array([0.0, 0.02]),
            "prior_wt": np.ones(NSAMP),
            "pixels": pe_pixels,
        },
        "sel": {
            "m1det": np.linspace(34.0, 40.0, N_SEL),
            "m2det": 0.8 * np.linspace(34.0, 40.0, N_SEL),
            "dL": np.linspace(430.0, 530.0, N_SEL),
            "chieff": np.zeros(N_SEL),
            "prior_wt": np.ones(N_SEL),
            "pixels": sel_pixels,
        },
    }


def _cell_arrays(cell):
    if cell == "plain_full":
        return _base_arrays()
    if cell == "plain_compact":
        return _base_arrays(compact=True)
    if cell in ("complete_volume", "complete_zero"):
        return _base_arrays(sparse=True)
    if cell == "bright":
        return _base_arrays()
    raise KeyError(cell)


def _record(diag, *, full=None):
    event_ll = np.asarray(diag["event_log_evidence"], dtype=float)
    event_var = np.asarray(diag["event_mc_variance"], dtype=float)
    selection = float(diag["selection_log_correction"])
    assembled = selection + float(np.sum(event_ll))
    if full is None:
        full = float(diag["log_likelihood"])
    full = float(full)
    if not np.isclose(full, assembled, rtol=1e-12, atol=0.0):
        raise RuntimeError(
            f"full/assembled mismatch: full={full:.17g}, assembled={assembled:.17g}"
        )
    return {
        "event_log_evidence": event_ll.tolist(),
        "event_mc_variance": event_var.tolist(),
        "log_mu": float(diag["log_mu"]),
        "n_eff": float(diag["n_eff"]),
        "selection_log_correction": selection,
        "assembled_from_parts": assembled,
        "full_log_likelihood": full,
    }


def _candidate():
    import jax
    import jax.numpy as jnp

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
    )
    from darksirens.population import get_fixed_population_params, pop_model_prior_parser

    cosmo = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
    params = CatalogParameters(n0=1.0e-2, delta=0.0, sigma_kde=0.0, z_depth=None)
    points = _population_points(pop_model_prior_parser, get_fixed_population_params)
    rows = {}

    for cell in CELLS:
        data = _cell_arrays(cell)
        catalog = GalaxyCatalog(
            apix=APIX,
            zgals=jnp.asarray(data["zgals"]),
            dzgals=jnp.asarray(data["dzgals"]),
            wgals=jnp.asarray(data["wgals"]),
            ngals=jnp.asarray(data["ngals"]),
            unique_pixels=None if data["unique_pixels"] is None else jnp.asarray(data["unique_pixels"]),
        )
        pe = make_gw_event(**data["pe"])
        sel = make_gw_event(**data["sel"])
        cache = None
        if cell in ("plain_full", "plain_compact"):
            cache = build_observed_density_cache(catalog)

        cell_rows = {}
        for point_name, theta_np in points:
            theta = jnp.asarray(theta_np)
            if cell in ("plain_full", "plain_compact"):
                d = dark_siren_log_likelihood(
                    cosmo, params, theta, pe, catalog, cache, sel, catalog, cache,
                    N_EVENTS, NSAMP, N_DRAW, pop_model=POP_MODEL,
                    max_likelihood_variance=MAX_VARIANCE, return_diagnostics=True,
                )
            elif cell in ("complete_volume", "complete_zero"):
                d = complete_catalog_siren_log_likelihood(
                    cosmo, params, theta, pe, catalog, sel, catalog,
                    N_EVENTS, NSAMP, N_DRAW, pop_model=POP_MODEL,
                    empty_policy="volume" if cell == "complete_volume" else "zero",
                    max_likelihood_variance=MAX_VARIANCE, return_diagnostics=True,
                )
            else:
                d = bright_siren_log_likelihood(
                    cosmo, theta, pe, catalog,
                    (Counterpart(z=0.10, dz=0.02, pixel=7, sky_marginalized=False),),
                    sel, N_EVENTS, NSAMP, N_DRAW, pop_model=POP_MODEL,
                    max_likelihood_variance=MAX_VARIANCE, return_diagnostics=True,
                )
            cell_rows[point_name] = _record(d._asdict())
        rows[cell] = cell_rows

    return rows


def _legacy():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
    from darksirens.gw.populations import pop_model_parser, pop_model_prior_parser
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.inference.utils import log_sample_weight
    from darksirens.likelihood.core import darksiren_log_likelihood
    from darksirens.likelihood.events import make_gw_event
    from darksirens.likelihood.selection import (
        compute_selection_term,
        log_evidence_and_mc_variance,
        selection_log_correction,
    )
    from darksirens.redshift import zgrid as prior_zgrid
    from darksirens.redshift.prior import eval_redshift_prior_with_state, prepare_redshift_prior_state
    from darksirens.utils.cosmology import dL_of_z, zgrid as distance_zgrid

    cosmo = CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
    points = _population_points(pop_model_prior_parser, get_fixed_population_params)
    rows = {}

    def make_catalog(data, cell):
        delta_g = jnp.zeros((data["zgals"].shape[0], int(prior_zgrid.shape[0])))
        kwargs = {}
        if cell == "bright":
            kwargs.update(
                counterpart_pixel=7,
                counterpart_pixels=jnp.array([7], dtype=jnp.int32),
                counterpart_zs=jnp.array([0.10]),
                counterpart_dzs=jnp.array([0.02]),
                bright_siren_sky_marginalized=False,
            )
        return EMCatalog(
            apix=APIX,
            zgals=jnp.asarray(data["zgals"]),
            dzgals=jnp.asarray(data["dzgals"]),
            wgals=jnp.asarray(data["wgals"]),
            ngals=jnp.asarray(data["ngals"]),
            delta_g_pix_z=delta_g,
            dN_obs_kde=None,
            pixel_to_cache_idx=None,
            unique_pixels=None if data["unique_pixels"] is None else jnp.asarray(data["unique_pixels"]),
            **kwargs,
        )

    for cell in CELLS:
        data = _cell_arrays(cell)
        policy = 1 if cell == "complete_volume" else 0
        survey = SurveyParams(
            n0=1.0e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
            alpha_miss=1.0, sigma_kde=0.0, complete_empty_pixel_policy=policy,
        )
        catalog = make_catalog(data, cell)
        pe = make_gw_event(**data["pe"])
        sel = make_gw_event(**data["sel"])

        if cell in ("plain_full", "plain_compact"):
            pe_model = sel_model = "dark_sirens"
        elif cell in ("complete_volume", "complete_zero"):
            pe_model = sel_model = "dark_sirens_complete"
        else:
            pe_model, sel_model = "bright_sirens", "spectral_sirens"

        cell_rows = {}
        for point_name, theta_np in points:
            theta = jnp.asarray(theta_np)
            log_p_pop = pop_model_parser(
                POP_MODEL, shared_beta=True, shared_spin=True, shared_gamma=True
            )
            dL_grid = dL_of_z(distance_zgrid, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
            dL_lo, dL_hi = dL_grid[0], dL_grid[-1]
            state_pe = (
                None if pe_model == "bright_sirens" else
                prepare_redshift_prior_state(pe_model, cosmo, survey, catalog, materialize_state=False)
            )
            state_sel = prepare_redshift_prior_state(
                sel_model, cosmo, survey, catalog, materialize_state=False
            )

            def make_weight(model, state, cat):
                def prior(z, pix, em):
                    return eval_redshift_prior_with_state(model, state, z, pix, cosmo, survey, em)

                def weight(m1det, q, dL, chieff, pix, prior_wt, _catalog=None, spin=None):
                    supported = (dL >= dL_lo) & (dL <= dL_hi)
                    dL_c = jnp.clip(dL, dL_lo, dL_hi)
                    ldw = log_sample_weight(
                        m1det, q, dL_c, chieff, pix, prior_wt,
                        cosmo, survey, theta, cat, log_p_pop, prior,
                        spin=spin, dL_grid=dL_grid,
                    )
                    return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

                return weight

            pe_weight = make_weight(pe_model, state_pe, catalog)
            sel_weight = make_weight(sel_model, state_sel, catalog)
            log_mu, n_eff, _ = compute_selection_term(sel, catalog, sel_weight, N_DRAW, N_EVENTS)

            ldw = pe_weight(
                pe.m1det, pe.q, pe.dL, pe.chieff, pe.pixels, pe.prior_wt,
                catalog, spin=pe.spin,
            )
            ldw = jnp.where(
                pe.valid & (pe.prior_wt > 0.0) & jnp.isfinite(ldw), ldw, -jnp.inf
            ).reshape(N_EVENTS, NSAMP)
            event_ll, event_var = jax.vmap(
                lambda row: log_evidence_and_mc_variance(row, NSAMP)
            )(ldw)
            selection_ll = selection_log_correction(
                log_mu, n_eff, N_EVENTS,
                max_likelihood_variance=MAX_VARIANCE,
                pe_variance_sum=jnp.sum(event_var),
            )
            assembled = selection_ll + jnp.sum(event_ll)
            full = darksiren_log_likelihood(
                cosmo, survey, theta, pe, catalog, sel, catalog,
                N_EVENTS, NSAMP, N_DRAW, POP_MODEL, pe_model,
                max_likelihood_variance=MAX_VARIANCE,
            )
            diag = {
                "log_likelihood": assembled,
                "event_log_evidence": event_ll,
                "event_mc_variance": event_var,
                "log_mu": log_mu,
                "n_eff": n_eff,
                "selection_log_correction": selection_ll,
            }
            cell_rows[point_name] = _record(diag, full=float(full))
        rows[cell] = cell_rows

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = _legacy() if args.implementation == "legacy" else _candidate()
    payload = {
        "implementation": args.implementation,
        "reference_sha": REFERENCE_SHA,
        "pop_model": POP_MODEL,
        "cells": list(CELLS),
        "point_names": list(POINT_NAMES),
        "point_fractions": list(POINT_FRACTIONS),
        "fixture": {
            "n_events": N_EVENTS,
            "nsamp": NSAMP,
            "n_found_injections": N_SEL,
            "n_draw": N_DRAW,
            "max_likelihood_variance": MAX_VARIANCE,
            "apix": APIX,
        },
        "results": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
