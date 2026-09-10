#!/usr/bin/env python3
"""Serialize deterministic cosmology outputs for legacy/new parity checks."""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

parser = argparse.ArgumentParser()
parser.add_argument("--api", choices=("legacy", "new"), required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

if args.api == "legacy":
    from darksirens.utils import cosmology as c
    from darksirens.redshift import grid as rg
else:
    from darksirens.cosmology import distances as c
    from darksirens.cosmology import _grid as rg

z = jnp.asarray([0.0, 1.0e-5, 5.0e-4, 0.01, 0.1, 0.3, 1.0, 4.5])
cosmologies = [
    (67.74, 0.3075, -1.0, 0.0),
    (80.0, 0.25, -0.7, 0.8),
    (55.0, 0.40, -1.8, -1.5),
]

out = {
    "jax": jax.__version__,
    "backend": jax.default_backend(),
    "distance_grid_shape": list(c.rs.shape),
    "distance_zgrid_shape": list(c.zgrid.shape),
    "prior_zgrid_shape": list(rg.zgrid.shape),
    "distance_zgrid_edges": [float(c.zgrid[0]), float(c.zgrid[-1])],
    "prior_zgrid_edges": [float(rg.zgrid[0]), float(rg.zgrid[-1])],
    "distance_table_samples": [
        float(c.rs[10, 20, 15, 0]),
        float(c.rs[10, 20, 15, 1]),
        float(c.rs[10, 20, 15, 100]),
        float(c.rs[-1, -1, -1, -1]),
    ],
    "cases": [],
}

for H0, Om0, w0, wa in cosmologies:
    dL = c.dL_of_z(z, H0, Om0, w0, wa)
    r = c.r_of_z(z, H0, Om0, w0, wa)
    dV = c.dV_of_z(z, H0, Om0, w0, wa)
    expansion = c.E(z, Om0, w0, wa)
    ddL = c.ddL_of_z(z, dL, H0, Om0, w0, wa)
    inv = c.z_of_dL(dL, H0, Om0, w0, wa)
    dm = c.distance_modulus(z, H0, Om0, w0, wa)
    out["cases"].append(
        {
            "parameters": [H0, Om0, w0, wa],
            "r": np.asarray(r).tolist(),
            "dL": np.asarray(dL).tolist(),
            "dV": np.asarray(dV).tolist(),
            "E": np.asarray(expansion).tolist(),
            "ddL": np.asarray(ddL).tolist(),
            "inverse_z": np.asarray(inv).tolist(),
            "distance_modulus": np.asarray(dm).tolist(),
        }
    )

# Probe the shared redshift grid away from nodes and around boundaries.
probe_z = jnp.asarray([-0.1, 0.0, 1e-8, 0.01, 0.12345, 1.0, float(rg.zgrid[-1]), 10.0])
log_grid = jnp.log(jnp.maximum(rg.zgrid, jnp.finfo(rg.zgrid.dtype).tiny) ** 2 + 1e-300)
out["prior_grid_probe"] = {
    "upper_index": np.asarray(rg.zgrid_upper_index(probe_z)).tolist(),
    "log_interp": np.asarray(rg.log_interp_zgrid(probe_z, log_grid)).tolist(),
}

with open(args.output, "w") as f:
    json.dump(out, f, indent=2, sort_keys=True, allow_nan=True)
