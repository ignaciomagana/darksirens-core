#!/usr/bin/env python3
"""Phase 5D1 separate-process parity probe for generic marked hosts."""

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
CELLS = ("marks",)
NPIX = 12
APIX = float(np.pi / 3.0)
MARK_ABS_MAX = 0.9
ETA_BOUND = 5.0


def _population_points(pop_model_prior_parser, get_fixed_population_params):
    lower, upper, labels, *_ = pop_model_prior_parser(POP_MODEL)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    fid = np.asarray(get_fixed_population_params(POP_MODEL), dtype=np.float64)
    if not labels:
        raise RuntimeError("population parser returned no labels")
    rows = []
    for name, frac in zip(POINT_NAMES, POINT_FRACTIONS):
        theta = fid.copy()
        theta[0] = lower[0] + frac * (upper[0] - lower[0])
        eta = -ETA_BOUND + frac * (2.0 * ETA_BOUND)
        rows.append((name, theta, eta))
    return rows


def _mark_table():
    col = MARK_ABS_MAX * np.linspace(-1.0, 1.0, NPIX)
    return np.stack([col, -col], axis=1).astype(np.float64)


def _arrays():
    """Reproduce the frozen factory's flat full-catalog UNION path.

    The Phase-1 ``marks`` golden starts from 12 full-sky rows, but
    ``prepare_catalog_views`` gathers one shared PE-union-selection table before
    the marked prior is built.  The only visited global pixels are 2 and 7, so
    the runtime catalog/mark rows are ``full[[2, 7]]`` and sample pixels are the
    corresponding compact row ids.  Using all 12 rows here changes
    ``mu_miss=E_obs[h|z]`` when eta != 0 and therefore probes a different marked
    state even though eta=0 is unchanged.
    """

    full_zgals = np.tile(np.array([0.06, 0.16], dtype=np.float64), (NPIX, 1))
    full_dzgals = np.full((NPIX, 2), 0.02, dtype=np.float64)
    full_wgals = np.ones((NPIX, 2), dtype=np.float64)
    full_ngals = np.full(NPIX, 2, dtype=np.int32)
    full_marks = _mark_table()

    pe_global = np.array([7, 7], dtype=np.int32)
    sel_global = np.array([2, 7, 2, 7, 2, 7, 2, 7], dtype=np.int32)
    union = np.unique(np.concatenate([pe_global, sel_global])).astype(np.int32)
    pe_rows = np.searchsorted(union, pe_global).astype(np.int32)
    sel_rows = np.searchsorted(union, sel_global).astype(np.int32)

    return {
        "union_pixels": union,
        "zgals": full_zgals[union],
        "dzgals": full_dzgals[union],
        "wgals": full_wgals[union],
        "ngals": full_ngals[union],
        "marks": full_marks[union],
        "pe": {
            "m1det": np.array([36.0, 38.0]),
            "m2det": np.array([28.8, 30.4]),
            "dL": np.array([460.0, 500.0]),
            "chieff": np.array([0.0, 0.02]),
            "prior_wt": np.ones(NSAMP),
            "pixels": pe_rows,
        },
        "sel": {
            "m1det": np.linspace(34.0, 40.0, N_SEL),
            "m2det": 0.8 * np.linspace(34.0, 40.0, N_SEL),
            "dL": np.linspace(430.0, 530.0, N_SEL),
            "chieff": np.zeros(N_SEL),
            "prior_wt": np.ones(N_SEL),
            "pixels": sel_rows,
        },
    }


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
    from darksirens.catalog.hosts import (
        CenteredHostMarks,
        LogLinearHostModel,
        check_centered_marks,
    )
    from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
    from darksirens.cosmology.parameters import CosmologyParameters
    from darksirens.gw.runtime import make_gw_event
    from darksirens.likelihood.marked import marked_dark_siren_log_likelihood
    from darksirens.population import get_fixed_population_params, pop_model_prior_parser

    data = _arrays()
    catalog = GalaxyCatalog(
        apix=APIX,
        zgals=jnp.asarray(data["zgals"]),
        dzgals=jnp.asarray(data["dzgals"]),
        wgals=jnp.asarray(data["wgals"]),
        ngals=jnp.asarray(data["ngals"]),
        unique_pixels=jnp.asarray(data["union_pixels"], dtype=jnp.int32),
    )
    marks = CenteredHostMarks(
        ("logmstar",), jnp.asarray(data["marks"])[..., None]
    )
    host_model = LogLinearHostModel(("logmstar",))
    check_centered_marks(host_model, marks, catalog, where="5D1 marks fixture")
    cache = build_observed_density_cache(catalog)
    pe = make_gw_event(**data["pe"])
    sel = make_gw_event(**data["sel"])
    cosmo = CosmologyParameters(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
    params = CatalogParameters(n0=1.0e-2, delta=0.0, sigma_kde=0.0, z_depth=None)

    rows = {}
    for point_name, theta_np, eta in _population_points(
        pop_model_prior_parser, get_fixed_population_params
    ):
        d = marked_dark_siren_log_likelihood(
            cosmo,
            params,
            jnp.asarray(theta_np),
            pe,
            catalog,
            cache,
            host_model,
            marks,
            jnp.asarray([eta]),
            sel,
            catalog,
            cache,
            marks,
            N_EVENTS,
            NSAMP,
            N_DRAW,
            pop_model=POP_MODEL,
            max_likelihood_variance=MAX_VARIANCE,
            return_diagnostics=True,
        )
        rows[point_name] = _record(d._asdict())
    return {"marks": rows}


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
    from darksirens.redshift.prior import (
        eval_redshift_prior_with_state,
        prepare_redshift_prior_state,
    )
    from darksirens.utils.cosmology import dL_of_z, zgrid as distance_zgrid

    data = _arrays()
    catalog = EMCatalog(
        apix=APIX,
        zgals=jnp.asarray(data["zgals"]),
        dzgals=jnp.asarray(data["dzgals"]),
        wgals=jnp.asarray(data["wgals"]),
        ngals=jnp.asarray(data["ngals"]),
        delta_g_pix_z=jnp.zeros((len(data["union_pixels"]), int(prior_zgrid.shape[0]))),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        unique_pixels=jnp.asarray(data["union_pixels"], dtype=jnp.int32),
        mark_logmstar=jnp.asarray(data["marks"]),
    )
    survey = SurveyParams(
        n0=1.0e-2,
        z50=1.0,
        w=0.5,
        delta=0.0,
        b_miss=1.0,
        alpha_miss=1.0,
        sigma_kde=0.0,
    )
    cosmo = CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)
    pe = make_gw_event(**data["pe"])
    sel = make_gw_event(**data["sel"])
    rows = {}

    for point_name, theta_np, eta in _population_points(
        pop_model_prior_parser, get_fixed_population_params
    ):
        theta = jnp.asarray(theta_np)
        eta_arr = jnp.asarray([eta])
        log_p_pop = pop_model_parser(
            POP_MODEL, shared_beta=True, shared_spin=True, shared_gamma=True
        )
        dL_grid = dL_of_z(
            distance_zgrid, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
        )
        dL_lo, dL_hi = dL_grid[0], dL_grid[-1]
        state_pe = prepare_redshift_prior_state(
            "dark_sirens",
            cosmo,
            survey,
            catalog,
            mark_model="loglinear",
            mark_params=eta_arr,
            mark_names=("logmstar",),
            materialize_state=False,
        )
        state_sel = prepare_redshift_prior_state(
            "dark_sirens",
            cosmo,
            survey,
            catalog,
            mark_model="loglinear",
            mark_params=eta_arr,
            mark_names=("logmstar",),
            materialize_state=False,
        )

        def make_weight(state):
            def prior(z, pix, em):
                return eval_redshift_prior_with_state(
                    "dark_sirens", state, z, pix, cosmo, survey, em
                )

            def weight(m1det, q, dL, chieff, pix, prior_wt, _catalog=None, spin=None):
                supported = (dL >= dL_lo) & (dL <= dL_hi)
                dL_c = jnp.clip(dL, dL_lo, dL_hi)
                ldw = log_sample_weight(
                    m1det,
                    q,
                    dL_c,
                    chieff,
                    pix,
                    prior_wt,
                    cosmo,
                    survey,
                    theta,
                    catalog,
                    log_p_pop,
                    prior,
                    spin=spin,
                    dL_grid=dL_grid,
                )
                return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

            return weight

        pe_weight = make_weight(state_pe)
        sel_weight = make_weight(state_sel)
        log_mu, n_eff, _ = compute_selection_term(
            sel, catalog, sel_weight, N_DRAW, N_EVENTS
        )
        ldw = pe_weight(
            pe.m1det,
            pe.q,
            pe.dL,
            pe.chieff,
            pe.pixels,
            pe.prior_wt,
            catalog,
            spin=pe.spin,
        )
        ldw = jnp.where(
            pe.valid & (pe.prior_wt > 0.0) & jnp.isfinite(ldw),
            ldw,
            -jnp.inf,
        ).reshape(N_EVENTS, NSAMP)
        event_ll, event_var = jax.vmap(
            lambda row: log_evidence_and_mc_variance(row, NSAMP)
        )(ldw)
        selection_ll = selection_log_correction(
            log_mu,
            n_eff,
            N_EVENTS,
            max_likelihood_variance=MAX_VARIANCE,
            pe_variance_sum=jnp.sum(event_var),
        )
        assembled = selection_ll + jnp.sum(event_ll)
        full = darksiren_log_likelihood(
            cosmo,
            survey,
            theta,
            pe,
            catalog,
            sel,
            catalog,
            N_EVENTS,
            NSAMP,
            N_DRAW,
            POP_MODEL,
            "dark_sirens",
            mark_model="loglinear",
            mark_params=eta_arr,
            mark_names=("logmstar",),
            max_likelihood_variance=MAX_VARIANCE,
        )
        rows[point_name] = _record(
            {
                "log_likelihood": assembled,
                "event_log_evidence": event_ll,
                "event_mc_variance": event_var,
                "log_mu": log_mu,
                "n_eff": n_eff,
                "selection_log_correction": selection_ll,
            },
            full=float(full),
        )
    return {"marks": rows}


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
            "union_pixels": [2, 7],
            "mark_names": ["logmstar"],
            "mark_abs_max": MARK_ABS_MAX,
            "eta_bound": ETA_BOUND,
        },
        "results": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
