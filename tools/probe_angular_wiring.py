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
    # One structurally invalid sample (first event) and one zero-prior-weight
    # sample (last event), so the structural PE sample mask is exercised.
    valid = np.ones(n, dtype=bool)
    valid[1] = False
    prior_wt = np.ones(n, dtype=np.float64)
    prior_wt[n - 2] = 0.0
    return SimpleNamespace(
        m1det=np.linspace(30.0, 42.0, n, dtype=np.float64),
        q=np.linspace(0.65, 0.9, n, dtype=np.float64),
        dL=np.asarray(dL, dtype=np.float64),
        chieff=np.zeros(n, dtype=np.float64),
        pixels=np.zeros(n, dtype=np.int32),
        prior_wt=prior_wt,
        valid=valid,
        nx=nx,
        ny=ny,
        nz=nz,
        spin=None,
    )


def _base_weight(m1det, q, dL, chieff, pix, prior_wt, catalog=None, spin=None):
    """Non-constant population stand-in with finite and -inf entries."""
    import jax.numpy as jnp

    del chieff, pix, prior_wt, catalog, spin
    finite = -0.5 * ((m1det - 36.0) / 3.0) ** 2 + jnp.log(q)
    return jnp.where(dL < 880.0, finite, -jnp.inf)


def _frozen_pe_reduction():
    """Return the frozen catalog-free PE reduction, executed from frozen source.

    The per-event PE reduction in the frozen reference is a closure inside
    ``darksiren_log_likelihood`` (``_pe_chunk_ldw`` plus the block-vectorized
    branch of ``if has_counterpart``).  Those statements are taken verbatim
    from the imported frozen ``darksirens.likelihood.core`` and run against
    that module's own globals; only the enclosing closure variables are
    supplied as arguments.
    """
    import ast

    import darksirens.likelihood.core as core

    source = Path(core.__file__)
    tree = ast.parse(source.read_text(), filename=str(source))

    def _one(nodes, pred, what):
        found = [node for node in nodes if pred(node)]
        if len(found) != 1:
            raise RuntimeError(f"expected one {what} in {source}; found {len(found)}")
        return found[0]

    def _func(nodes, name):
        return _one(
            nodes,
            lambda n: isinstance(n, ast.FunctionDef) and n.name == name,
            f"def {name}",
        )

    outer = _func(tree.body, "darksiren_log_likelihood")
    inner = _func(outer.body, "_ll_given_states")
    chunk = _func(inner.body, "_pe_chunk_ldw")
    branch = _one(
        inner.body,
        lambda n: isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "has_counterpart",
        "if has_counterpart",
    )
    template = ast.parse(
        "def frozen_pe_reduction(gw_pe, nEvents, nsamp, pe_block, log_weight_ev,"
        " catalogs_pe_all, frozen, frozen_prior, _frozen_mix, apply_sky,"
        " log_g_sky, sky_params, _dL_lo, _dL_hi, _dL_grid):\n"
        "    return event_lls, event_vars\n"
    )
    fn = template.body[0]
    fn.body = [chunk] + list(branch.orelse) + fn.body
    module = ast.fix_missing_locations(template)
    namespace = dict(vars(core))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["frozen_pe_reduction"]


def _legacy():
    import jax.numpy as jnp
    from darksirens.likelihood.selection import compute_selection_term
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

    def log_weight_ev(m1det, q, dL, chieff, pix, prior_wt, catalogs, spin=None,
                      log_prior_vals=None):
        assert log_prior_vals is None
        return _base_weight(m1det, q, dL, chieff, pix, prior_wt, catalogs, spin=spin)

    pe = _event(8, np.linspace(350.0, 900.0, 8))
    pe = SimpleNamespace(**{k: jnp.asarray(v) if v is not None else None for k, v in vars(pe).items()})
    event_ll, event_var = _frozen_pe_reduction()(
        gw_pe=pe,
        nEvents=2,
        nsamp=4,
        pe_block=2,
        log_weight_ev=log_weight_ev,
        catalogs_pe_all=(None,),
        frozen=False,
        frozen_prior=None,
        _frozen_mix=None,
        apply_sky=True,
        log_g_sky=log_g,
        sky_params=theta,
        _dL_lo=lo,
        _dL_hi=hi,
        _dL_grid=dL_grid,
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
