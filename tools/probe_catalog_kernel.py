#!/usr/bin/env python3
"""Serialize the same ordinary catalog-kernel fixture from legacy or candidate."""

from __future__ import annotations

import argparse
import json

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--implementation", choices=("legacy", "candidate"), required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()


ZGALS = jnp.asarray(
    [
        [0.045, 0.18, 0.41, 100.0],
        [0.31, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0],
    ]
)
DZGALS = jnp.asarray(
    [
        [0.004, 0.035, 0.08, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
)
WGALS = jnp.asarray(
    [
        [1.0, 2.5, 0.3, 0.0],
        [4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
)
NGALS = jnp.asarray([3, 1, 0], dtype=jnp.int32)
UNIQUE_PIXELS = jnp.asarray([7, 1_000_003, 9_000_017], dtype=jnp.int32)

QUERY_Z = jnp.asarray([0.012, 0.047, 0.17, 0.32, 0.62, 1.25])
QUERY_ROW = jnp.asarray([0, 0, 0, 1, 0, 1], dtype=jnp.int32)
DEPTH = 0.28
DEPTH_Z = jnp.asarray([0.03, 0.12, 0.22, 0.27])
DEPTH_ROW = jnp.zeros(DEPTH_Z.shape, dtype=jnp.int32)

POINTS = (
    dict(H0=55.0, delta=0.0, sigma_kde=0.0),
    dict(H0=67.74, delta=0.6, sigma_kde=0.015),
    dict(H0=95.0, delta=-0.3, sigma_kde=0.035),
)


if args.implementation == "legacy":
    from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
    from darksirens.redshift import zgrid
    from darksirens.redshift.catalog import (
        catalog_kernel_state,
        eval_log_catalog_prior_state,
        log_catalog_prior_vmap,
    )
    from darksirens.redshift.completion import log_galaxy_measure_grid

    catalog = EMCatalog(
        apix=1.0,
        zgals=ZGALS,
        dzgals=DZGALS,
        wgals=WGALS,
        ngals=NGALS,
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        unique_pixels=UNIQUE_PIXELS,
    )

    def make_cosmo(point):
        return CosmoParams(H0=point["H0"], Om0=0.3075, w0=-1.0, wa=0.0)

    def make_params(point, depth=None):
        return SurveyParams(
            n0=1.0,
            z50=1.0,
            w=0.5,
            delta=point["delta"],
            b_miss=1.0,
            alpha_miss=1.0,
            sigma_kde=point["sigma_kde"],
            z_depth=depth,
        )

    def build_state(cosmo, params, depth=None):
        return catalog_kernel_state(cosmo, params, catalog, z_depth=depth)

    def direct(z, row, cosmo, params):
        return log_catalog_prior_vmap(z, row, cosmo, params, catalog)

    def evaluate(z, row, state):
        return jax.vmap(
            lambda zi, ri: eval_log_catalog_prior_state(zi, ri, state, catalog)
        )(z, row)

else:
    from darksirens.catalog.redshift import (
        build_catalog_kernel_state,
        eval_log_catalog_prior_state_vmap,
        log_catalog_prior_vmap,
        log_galaxy_measure_grid,
    )
    from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
    from darksirens.cosmology._grid import zgrid
    from darksirens.cosmology.parameters import CosmologyParameters

    catalog = GalaxyCatalog(
        apix=1.0,
        zgals=ZGALS,
        dzgals=DZGALS,
        wgals=WGALS,
        ngals=NGALS,
        unique_pixels=UNIQUE_PIXELS,
    )

    def make_cosmo(point):
        return CosmologyParameters(
            H0=point["H0"], Om0=0.3075, w0=-1.0, wa=0.0
        )

    def make_params(point, depth=None):
        return CatalogParameters(
            n0=1.0,
            delta=point["delta"],
            sigma_kde=point["sigma_kde"],
            z_depth=depth,
        )

    def build_state(cosmo, params, depth=None):
        return build_catalog_kernel_state(cosmo, params, catalog)

    def direct(z, row, cosmo, params):
        return log_catalog_prior_vmap(z, row, cosmo, params, catalog)

    def evaluate(z, row, state):
        return eval_log_catalog_prior_state_vmap(z, row, state, catalog)


def as_list(value):
    return np.asarray(value).tolist()


records = []
for point in POINTS:
    cosmo = make_cosmo(point)
    params = make_params(point)
    state = build_state(cosmo, params)
    direct_values = direct(QUERY_Z, QUERY_ROW, cosmo, params)
    state_values = evaluate(QUERY_Z, QUERY_ROW, state)

    depth_params = make_params(point, DEPTH)
    depth_state = build_state(cosmo, depth_params, depth=DEPTH)
    depth_values = evaluate(DEPTH_Z, DEPTH_ROW, depth_state)
    above = evaluate(
        jnp.asarray([DEPTH + 0.01, DEPTH + 0.4]),
        jnp.zeros(2, dtype=jnp.int32),
        depth_state,
    )

    empty = evaluate(
        jnp.asarray([0.1, 0.4]),
        jnp.full(2, 2, dtype=jnp.int32),
        state,
    )
    grid_idx = np.asarray(
        [1, len(zgrid) // 10, len(zgrid) // 2, len(zgrid) - 1], dtype=int
    )
    log_g = log_galaxy_measure_grid(cosmo, params)

    records.append(
        {
            "point": point,
            "log_g_samples": as_list(np.asarray(log_g)[grid_idx]),
            "log_kw": as_list(state.log_kw),
            "sig_eff": as_list(state.sig_eff),
            "row_empty": as_list(state.row_empty),
            "direct_logp": as_list(direct_values),
            "state_logp": as_list(state_values),
            "depth_log_kw": as_list(depth_state.log_kw),
            "depth_log_mass": as_list(depth_state.log_depth_mass),
            "depth_logp_below": as_list(depth_values),
            "depth_above_is_neginf": bool(np.all(np.isneginf(np.asarray(above)))),
            "empty_is_neginf": bool(np.all(np.isneginf(np.asarray(empty)))),
        }
    )

payload = {
    "implementation": args.implementation,
    "zgrid_size": int(len(zgrid)),
    "quadrature_nodes": 24,
    "records": records,
}
with open(args.out, "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
print(args.out)
