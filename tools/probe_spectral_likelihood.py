#!/usr/bin/env python3
"""Evaluate the same fixed-theta spectral-siren fixture in legacy or core.

Run this script in separate Python processes with either the pinned legacy root
or ``darksirens-core/src`` first on PYTHONPATH. The two implementations expose
the same package name and must never share an interpreter during parity tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REFERENCE_SHA = "c042527238bd71421b792936bc48c3b815b90d6d"
POP_MODEL = "powerlaw+peak"
N_EVENTS = 3
NSAMP = 8
N_SEL = 257
N_DRAW = 4096.0
SEL_BATCH = 64
MAX_VARIANCE = 1.0e6


def _fixture(make_gw_event):
    n_pe = N_EVENTS * NSAMP
    i_pe = np.arange(n_pe, dtype=float)
    m1_pe = np.linspace(31.0, 47.0, n_pe)
    q_pe = np.linspace(0.66, 0.93, n_pe)
    p_pe = 0.85 + 0.25 * (1.0 + np.sin(0.37 * i_pe)) / 2.0
    valid_pe = np.ones(n_pe, dtype=bool)
    valid_pe[[4, 17]] = False
    pe = make_gw_event(
        m1det=m1_pe,
        m2det=m1_pe * q_pe,
        dL=np.linspace(360.0, 1180.0, n_pe),
        chieff=np.linspace(-0.16, 0.21, n_pe),
        prior_wt=p_pe,
        pixels=np.zeros(n_pe, dtype=np.int32),
        valid=valid_pe,
    )

    i_sel = np.arange(N_SEL, dtype=float)
    m1_sel = np.linspace(27.5, 53.0, N_SEL)
    q_sel = np.linspace(0.58, 0.97, N_SEL)
    p_sel = 0.75 + 0.35 * (1.0 + np.cos(0.071 * i_sel)) / 2.0
    valid_sel = np.ones(N_SEL, dtype=bool)
    valid_sel[[13, 201]] = False
    sel = make_gw_event(
        m1det=m1_sel,
        m2det=m1_sel * q_sel,
        dL=np.linspace(260.0, 1650.0, N_SEL),
        chieff=np.linspace(-0.24, 0.28, N_SEL),
        prior_wt=p_sel,
        pixels=np.zeros(N_SEL, dtype=np.int32),
        valid=valid_sel,
    )
    return pe, sel


def _theta_points(get_fixed_population_params):
    fid = np.asarray(get_fixed_population_params(POP_MODEL), dtype=float)
    gamma = fid.copy()
    gamma[-1] = min(9.0, gamma[-1] + 0.4)
    mixture = fid.copy()
    mixture[0] = np.clip(0.82 * mixture[0] + 0.07, 0.05, 0.95)
    return [
        ("low_h0_fid", 55.0, fid),
        ("anchor_gamma_shift", 67.74, gamma),
        ("high_h0_mixture_shift", 90.0, mixture),
    ]


def _as_record(diag, *, assembled=None, full=None):
    event_ll = np.asarray(diag["event_log_evidence"], dtype=float)
    event_var = np.asarray(diag["event_mc_variance"], dtype=float)
    selection = float(diag["selection_log_correction"])
    if assembled is None:
        assembled = selection + float(np.sum(event_ll))
    if full is None:
        full = assembled
    if not np.isclose(float(full), float(assembled), rtol=1e-12, atol=0.0):
        raise RuntimeError(
            f"full/assembled mismatch: full={full:.17g}, assembled={assembled:.17g}"
        )
    return {
        "event_log_evidence": event_ll.tolist(),
        "event_mc_variance": event_var.tolist(),
        "log_mu": float(diag["log_mu"]),
        "n_eff": float(diag["n_eff"]),
        "selection_log_correction": selection,
        "assembled_from_parts": float(assembled),
        "full_log_likelihood": float(full),
    }


def _candidate():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from darksirens.cosmology.parameters import CosmologyParameters
    from darksirens.gw.runtime import make_gw_event
    from darksirens.likelihood.hierarchical import spectral_siren_log_likelihood
    from darksirens.population import get_fixed_population_params

    pe, sel = _fixture(make_gw_event)
    rows = {}
    for name, h0, theta in _theta_points(get_fixed_population_params):
        cosmo = CosmologyParameters(H0=h0, Om0=0.3075, w0=-1.0, wa=0.0)
        d = spectral_siren_log_likelihood(
            cosmo,
            jnp.asarray(theta),
            pe,
            sel,
            N_EVENTS,
            NSAMP,
            N_DRAW,
            pop_model=POP_MODEL,
            shared_beta=True,
            shared_spin=True,
            shared_gamma=True,
            sel_batch_size=SEL_BATCH,
            pe_event_block=None,
            max_likelihood_variance=MAX_VARIANCE,
            return_diagnostics=True,
        )
        rows[name] = _as_record({
            "event_log_evidence": d.event_log_evidence,
            "event_mc_variance": d.event_mc_variance,
            "log_mu": d.log_mu,
            "n_eff": d.n_eff,
            "selection_log_correction": d.selection_log_correction,
        }, full=float(d.log_likelihood))
    return rows


def _legacy():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
    from darksirens.gw.populations import pop_model_parser
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.inference.utils import log_sample_weight
    from darksirens.likelihood.core import darksiren_log_likelihood
    from darksirens.likelihood.events import make_gw_event
    from darksirens.likelihood.selection import (
        compute_selection_term,
        log_evidence_and_mc_variance,
        selection_log_correction,
    )
    from darksirens.redshift.grid import zgrid as redshift_grid
    from darksirens.redshift.volume import log_volume_prior
    from darksirens.utils.cosmology import dL_of_z, zgrid as distance_zgrid

    pe, sel = _fixture(make_gw_event)
    survey = SurveyParams(
        n0=1.0e-3,
        z50=0.3,
        w=0.05,
        delta=0.0,
        b_miss=1.0,
        alpha_miss=1.0,
    )
    catalog = EMCatalog(
        apix=1.0,
        zgals=jnp.zeros((1, 1)),
        dzgals=jnp.ones((1, 1)) * 0.01,
        wgals=jnp.ones((1, 1)),
        ngals=jnp.ones(1, dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, int(redshift_grid.shape[0]))),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )

    rows = {}
    for name, h0, theta_np in _theta_points(get_fixed_population_params):
        theta = jnp.asarray(theta_np)
        cosmo = CosmoParams(H0=h0, Om0=0.3075, w0=-1.0, wa=0.0)
        log_p_pop = pop_model_parser(
            POP_MODEL, shared_beta=True, shared_spin=True, shared_gamma=True
        )
        dL_grid = dL_of_z(
            distance_zgrid, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
        )
        dL_lo, dL_hi = dL_grid[0], dL_grid[-1]

        def log_prior_z(z, pix, cat):
            del pix, cat
            return jax.vmap(
                lambda zz: log_volume_prior(zz, cosmo, survey)
            )(z)

        def log_weight(m1det, q, dL, chieff, pix, prior_wt, cat, spin=None):
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
                cat,
                log_p_pop,
                log_prior_z,
                spin=spin,
                dL_grid=dL_grid,
            )
            return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

        log_mu, n_eff, _ = compute_selection_term(
            sel,
            catalog,
            log_weight,
            N_DRAW,
            N_EVENTS,
            sel_batch_size=SEL_BATCH,
        )

        if pe.spin is None:
            ldw = log_weight(
                pe.m1det, pe.q, pe.dL, pe.chieff, pe.pixels, pe.prior_wt, catalog
            )
        else:
            ldw = log_weight(
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
            "spectral_sirens",
            shared_beta=True,
            shared_spin=True,
            shared_gamma=True,
            sel_batch_size=SEL_BATCH,
            pe_event_block=None,
            max_likelihood_variance=MAX_VARIANCE,
        )
        rows[name] = _as_record({
            "event_log_evidence": event_ll,
            "event_mc_variance": event_var,
            "log_mu": log_mu,
            "n_eff": n_eff,
            "selection_log_correction": selection_ll,
        }, assembled=float(assembled), full=float(full))
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
        "fixture": {
            "n_events": N_EVENTS,
            "nsamp": NSAMP,
            "n_found_injections": N_SEL,
            "n_draw": N_DRAW,
            "selection_batch": SEL_BATCH,
            "max_likelihood_variance": MAX_VARIANCE,
        },
        "points": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
