#!/usr/bin/env python3
"""Serialize the same ordinary completeness fixture from legacy or candidate."""

from __future__ import annotations

import argparse
import json
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--implementation", choices=("legacy", "candidate"), required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

APIX = 8.0e-4
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

POINTS = (
    dict(H0=55.0, n0=1.0e-3, delta=0.0, z_depth=None),
    dict(H0=67.74, n0=2.0e-3, delta=0.6, z_depth=0.45),
    dict(H0=95.0, n0=7.0e-4, delta=-0.3, z_depth=None),
)

if args.implementation == "legacy":
    from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
    from darksirens.redshift import zgrid
    from darksirens.redshift.completion import (
        _S_EXP,
        _precompute_grids,
        _row_C,
        build_pixel_kde_cache,
        completion_curves,
    )

    catalog = EMCatalog(
        apix=APIX,
        zgals=ZGALS,
        dzgals=DZGALS,
        wgals=WGALS,
        ngals=NGALS,
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        unique_pixels=UNIQUE_PIXELS,
    )

    def make_cosmo(point):
        return CosmoParams(H0=point["H0"], Om0=0.3075, w0=-1.0, wa=0.0)

    def make_params(point):
        return SurveyParams(
            n0=point["n0"],
            z50=1.0,
            w=0.5,
            delta=point["delta"],
            b_miss=1.0,
            alpha_miss=1.0,
            sigma_kde=0.015,
            z_depth=point["z_depth"],
        )

    # Compact row ids are the cache keys.  The huge global ids remain metadata
    # only and must not force a dense max(pixel)+1 lookup.
    kde, _ = build_pixel_kde_cache(
        np.arange(ZGALS.shape[0], dtype=np.int32),
        ZGALS,
        ZGALS.shape[0],
        ngals=NGALS,
    )
    catalog = catalog._replace(dN_obs_kde=kde)
    smooth = _S_EXP

    def state(cosmo, params):
        return _precompute_grids(cosmo, params, catalog)

    def curves(cosmo, params):
        return completion_curves(cosmo, params, catalog)

    def get_log_g(grid):
        return grid.log_g

    def get_C(curve, grid):
        rows = jnp.arange(ZGALS.shape[0], dtype=jnp.int32)
        return jax.vmap(lambda row: _row_C(row, grid, catalog)[0])(rows)

else:
    from darksirens.catalog.completeness import (
        build_completion_state,
        build_observed_density_cache,
        completion_curves,
        smoothing_operator,
    )
    from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
    from darksirens.cosmology._grid import zgrid
    from darksirens.cosmology.parameters import CosmologyParameters

    catalog = GalaxyCatalog(
        apix=APIX,
        zgals=ZGALS,
        dzgals=DZGALS,
        wgals=WGALS,
        ngals=NGALS,
        unique_pixels=UNIQUE_PIXELS,
    )
    cache = build_observed_density_cache(catalog)
    smooth = smoothing_operator()

    def make_cosmo(point):
        return CosmologyParameters(
            H0=point["H0"], Om0=0.3075, w0=-1.0, wa=0.0
        )

    def make_params(point):
        return CatalogParameters(
            n0=point["n0"],
            delta=point["delta"],
            sigma_kde=0.015,
            z_depth=point["z_depth"],
        )

    def state(cosmo, params):
        return build_completion_state(cosmo, params, catalog)

    def curves(cosmo, params):
        return completion_curves(cosmo, params, catalog, cache)

    def get_log_g(grid):
        return grid.log_g_grid

    def get_C(curve, grid):
        return curve.C


def encode(value):
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, (np.floating, float)):
        x = float(value)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


# Probe the same sparse set of smoothing-matrix entries in addition to the
# end-to-end curves.  This catches a boundary-normalization or trapezoid-weight
# drift even if a particular count fixture does not excite it strongly.
idx = np.asarray([0, 1, 17, len(zgrid) // 3, len(zgrid) // 2, len(zgrid) - 1])
smooth_np = np.asarray(smooth)
smooth_probe = smooth_np[np.ix_(idx, idx)]

records = []
for point in POINTS:
    cosmo = make_cosmo(point)
    params = make_params(point)
    grid = state(cosmo, params)
    curve = curves(cosmo, params)
    C = get_C(curve, grid)

    records.append(
        {
            "point": point,
            "log_g": np.asarray(get_log_g(grid)),
            "dN_exp": np.asarray(grid.dN_exp),
            "dN_exp_smooth": np.asarray(grid.dN_exp_smooth),
            "C": np.asarray(C),
            "f": np.asarray(curve.f),
            "dN_miss": np.asarray(curve.dN_miss),
            "C_eff": np.asarray(curve.C_eff),
            "N_miss": np.asarray(curve.N_miss),
        }
    )

payload = encode(
    {
        "implementation": args.implementation,
        "zgrid_size": int(len(zgrid)),
        "sigma_smooth": 0.05,
        "smoothing_probe": smooth_probe,
        "records": records,
    }
)
with open(args.out, "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
print(args.out)
