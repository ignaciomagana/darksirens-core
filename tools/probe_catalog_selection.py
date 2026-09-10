#!/usr/bin/env python3
"""Serialize identical 5D2 selection-runtime fixtures from legacy or candidate."""

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
        [0.06, 0.19, 100.0],
        [0.13, 100.0, 100.0],
        [100.0, 100.0, 100.0],
    ]
)
DZGALS = jnp.asarray(
    [
        [0.004, 0.01, 1.0],
        [0.007, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
)
WGALS = jnp.asarray(
    [
        [1.0, 2.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
)
NGALS = jnp.asarray([2, 1, 0], dtype=jnp.int32)
UNIQUE_PIXELS = jnp.asarray([7, 1003, 9001], dtype=jnp.int32)

POINTS = (
    {
        "family": "gaussian",
        "H0": 55.0,
        "n0": 1.0e-3,
        "delta": 0.0,
        "z_depth": None,
        "m_lim": 24.0,
        "M0hat": -20.2,
        "sigma_M": 1.0,
        "k_corr_coeffs": (),
    },
    {
        "family": "gaussian",
        "H0": 95.0,
        "n0": 2.5e-3,
        "delta": 0.6,
        "z_depth": 0.45,
        "m_lim": 23.2,
        "M0hat": -20.7,
        "sigma_M": 0.82,
        "k_corr_coeffs": (0.39, 0.11),
    },
    {
        "family": "schechter",
        "H0": 67.74,
        "n0": 7.0e-4,
        "delta": -0.3,
        "z_depth": None,
        "m_lim": 21.5,
        "Mstar_hat": -20.31,
        "alpha": -0.7,
        "M_faint_offset": 5.0,
    },
    {
        "family": "schechter",
        "H0": 80.0,
        "n0": 1.7e-3,
        "delta": 0.2,
        "z_depth": 0.32,
        "m_lim": 22.0,
        "Mstar_hat": -20.5,
        "alpha": -1.21,
        "M_faint_offset": 4.5,
    },
)

if args.implementation == "legacy":
    from darksirens.core.types import (
        C_MODE_SELECTION_STRUCT,
        SELECTION_FAMILY_SCHECHTER_STRUCT,
        CosmoParams,
        EMCatalog,
        SurveyParams,
    )
    from darksirens.redshift import zgrid
    from darksirens.redshift.completion import _precompute_grids, _row_C, completion_curves
    from darksirens.redshift.selection import c_sel_gaussian, c_sel_schechter

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
        base = dict(
            n0=point["n0"],
            z50=1.0,
            w=0.5,
            delta=point["delta"],
            b_miss=1.0,
            alpha_miss=1.0,
            sigma_kde=0.0,
            z_depth=point["z_depth"],
            c_mode=C_MODE_SELECTION_STRUCT,
        )
        if point["family"] == "gaussian":
            base.update(
                m_lim=point["m_lim"],
                M0hat=point["M0hat"],
                sigma_M=point["sigma_M"],
                k_corr_coeffs=point["k_corr_coeffs"] or None,
            )
        else:
            base.update(
                selection_family=SELECTION_FAMILY_SCHECHTER_STRUCT,
                m_lim=point["m_lim"],
                Mstar_hat=point["Mstar_hat"],
                alpha=point["alpha"],
                M_faint_offset=point["M_faint_offset"],
            )
        return SurveyParams(**base)

    def model_curve(point, cosmo):
        if point["family"] == "gaussian":
            return c_sel_gaussian(
                zgrid,
                point["m_lim"],
                point["M0hat"],
                point["sigma_M"],
                cosmo.H0,
                cosmo.Om0,
                cosmo.w0,
                cosmo.wa,
                k_corr_coeffs=point["k_corr_coeffs"] or None,
            )
        return c_sel_schechter(
            zgrid,
            point["m_lim"],
            point["Mstar_hat"],
            point["alpha"],
            point["M_faint_offset"],
            cosmo.H0,
            cosmo.Om0,
            cosmo.w0,
            cosmo.wa,
        )

    def evaluate(point):
        cosmo = make_cosmo(point)
        survey = make_params(point)
        grids = _precompute_grids(cosmo, survey, catalog)
        curves = completion_curves(cosmo, survey, catalog)
        rows = jnp.arange(ZGALS.shape[0], dtype=jnp.int32)
        C = jax.vmap(lambda row: _row_C(row, grids, catalog)[0])(rows)
        return {
            "curve_raw": np.asarray(model_curve(point, cosmo)),
            "C": np.asarray(C),
            "dN_exp": np.asarray(grids.dN_exp),
            "f": np.asarray(curves.f),
            "dN_miss": np.asarray(curves.dN_miss),
            "C_eff": np.asarray(curves.C_eff),
            "N_miss": np.asarray(curves.N_miss),
        }

else:
    from darksirens.catalog.completeness import build_completion_state
    from darksirens.catalog.types import CatalogParameters, GalaxyCatalog
    from darksirens.cosmology._grid import zgrid
    from darksirens.cosmology.parameters import CosmologyParameters
    from darksirens.selection.catalog import (
        GaussianMagnitudeSelection,
        SchechterMagnitudeSelection,
        selection_completion_curves,
        selection_curve,
    )

    catalog = GalaxyCatalog(
        apix=APIX,
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

    def make_params(point):
        return CatalogParameters(
            n0=point["n0"],
            delta=point["delta"],
            sigma_kde=0.0,
            z_depth=point["z_depth"],
        )

    def make_model(point):
        if point["family"] == "gaussian":
            return GaussianMagnitudeSelection(
                point["m_lim"],
                point["M0hat"],
                point["sigma_M"],
                tuple(point["k_corr_coeffs"]),
            )
        return SchechterMagnitudeSelection(
            point["m_lim"],
            point["Mstar_hat"],
            point["alpha"],
            point["M_faint_offset"],
        )

    def evaluate(point):
        cosmo = make_cosmo(point)
        params = make_params(point)
        model = make_model(point)
        grids = build_completion_state(cosmo, params, catalog)
        curves = selection_completion_curves(cosmo, params, catalog, model)
        return {
            "curve_raw": np.asarray(selection_curve(zgrid, cosmo, model)),
            "C": np.asarray(curves.C),
            "dN_exp": np.asarray(grids.dN_exp),
            "f": np.asarray(curves.f),
            "dN_miss": np.asarray(curves.dN_miss),
            "C_eff": np.asarray(curves.C_eff),
            "N_miss": np.asarray(curves.N_miss),
        }


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


records = []
for point in POINTS:
    records.append({"point": point, **evaluate(point)})

payload = encode(
    {
        "implementation": args.implementation,
        "zgrid_size": int(len(zgrid)),
        "records": records,
    }
)
with open(args.out, "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
print(args.out)
