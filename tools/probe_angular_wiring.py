#!/usr/bin/env python3
"""Separate-process frozen/candidate dipole PE+selection wiring parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _hex_array(value):
    arr = np.asarray(value, dtype=np.float64)
    return [float(x).hex() for x in arr.reshape(-1)]


def _fixture(n):
    phi = np.linspace(0.15, 5.7, n, dtype=np.float64)
    zdir = np.linspace(-0.8, 0.8, n, dtype=np.float64)
    r = np.sqrt(1.0 - zdir * zdir)
    return r * np.cos(phi), r * np.sin(phi), zdir


def _event(n, dL):
    nx, ny, nz = _fixture(n)
    return SimpleNamespace(
        m1det=np.linspace(30.0, 42.0, n, dtype=np.float64),
        q=np.linspace(0.65, 0.9, n, dtype=np.float64),
        dL=np.asarray(dL, dtype=np.float64),
        chieff=np.zeros(n, dtype=np.float64),
        pixels=np.zeros(n, dtype=np.int32),
        prior_wt=np.ones(n, dtype=np.float64),
        valid=np.ones(n, dtype=bool),
        nx=nx,
        ny=ny,
        nz=nz,
        spin=None,
    )


def _base_weight(m1det, q, dL, chieff, pix, prior_wt, catalog=None, spin=None):
    del m1det, q, chieff, pix, prior_wt, catalog, spin
    return np.zeros_like(dL)


def _legacy():
    import jax.numpy as jnp
    from darksirens.likelihood.selection import (
        compute_selection_term,
        log_evidence_and_mc_variance,
    )
    from darksirens.sky import sky_model_parser
    from darksirens.utils.cosmology import (
        H0Planck,
        Om0Planck,
        dL_of_z,
        w0Fiducial,
        waFiducial,
        z_of_dL_precomputed,
        zgrid,
    )

    theta = jnp.asarray([0.2, -0.1, 0.05])
    log_g = sky_model_parser("dipole")
    dL_grid = dL_of_z(zgrid, H0Planck, Om0Planck, w0Fiducial, waFiducial)
    lo, hi = dL_grid[0], dL_grid[-1]

    def sky(nx, ny, nz, dL):
        z = z_of_dL_precomputed(jnp.clip(dL, lo, hi), dL_grid)
        return log_g(nx, ny, nz, z, theta)

    pe = _event(8, np.linspace(350.0, 900.0, 8))
    ldw = sky(
        jnp.asarray(pe.nx), jnp.asarray(pe.ny), jnp.asarray(pe.nz),
        jnp.asarray(pe.dL),
    ).reshape(2, 4)
    event = [log_evidence_and_mc_variance(row, 4) for row in ldw]
    event_ll = jnp.stack([x[0] for x in event])
    event_var = jnp.stack([x[1] for x in event])

    sel = _event(12, np.linspace(300.0, 1200.0, 12))
    sel = SimpleNamespace(**{k: jnp.asarray(v) if v is not None else None for k, v in vars(sel).items()})
    log_mu, neff, log_sigma2 = compute_selection_term(
        sel, None, _base_weight, 40.0, 2, sky_log_weight_fn=sky
    )
    return {
        "event_ll": _hex_array(event_ll),
        "event_var": _hex_array(event_var),
        "log_mu": float(log_mu).hex(),
        "neff": float(neff).hex(),
        "log_sigma2": float(log_sigma2).hex(),
    }


def _candidate():
    import jax.numpy as jnp
    from darksirens.cosmology.distances import (
        H0Planck,
        Om0Planck,
        dL_of_z,
        w0Fiducial,
        waFiducial,
        z_of_dL_precomputed,
        zgrid,
    )
    from darksirens.likelihood.event import reduce_pe_events
    from darksirens.population.angular import angular_model_parser
    from darksirens.selection.gw import compute_selection_term

    theta = jnp.asarray([0.2, -0.1, 0.05])
    log_g = angular_model_parser("dipole")
    dL_grid = dL_of_z(zgrid, H0Planck, Om0Planck, w0Fiducial, waFiducial)
    lo, hi = dL_grid[0], dL_grid[-1]

    def sky(nx, ny, nz, dL):
        z = z_of_dL_precomputed(jnp.clip(dL, lo, hi), dL_grid)
        return log_g(nx, ny, nz, z, theta)

    pe = _event(8, np.linspace(350.0, 900.0, 8))
    pe = SimpleNamespace(**{k: jnp.asarray(v) if v is not None else None for k, v in vars(pe).items()})
    event_ll, event_var = reduce_pe_events(
        pe, 2, 4, _base_weight, sky_log_weight_fn=sky
    )

    sel = _event(12, np.linspace(300.0, 1200.0, 12))
    sel = SimpleNamespace(**{k: jnp.asarray(v) if v is not None else None for k, v in vars(sel).items()})
    log_mu, neff, log_sigma2 = compute_selection_term(
        sel, None, _base_weight, 40.0, 2, sky_log_weight_fn=sky
    )
    return {
        "event_ll": _hex_array(event_ll),
        "event_var": _hex_array(event_var),
        "log_mu": float(log_mu).hex(),
        "neff": float(neff).hex(),
        "log_sigma2": float(log_sigma2).hex(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = _legacy() if args.mode == "legacy" else _candidate()
    Path(args.out).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
