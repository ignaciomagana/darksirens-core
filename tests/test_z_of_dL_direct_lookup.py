"""``z_of_dL`` lookup ``"direct"``: the distance-table bracket by index arithmetic.

The default (``"search"``) inverts the per-proposal ``dL_grid`` with an
unrolled binary search over the 500 table nodes for every PE sample and
injection. The opt-in ``"direct"`` lookup maps ``log(dL)`` to a cell of a
per-proposal table, corrects the bracket by one node on either side, and
applies the default's interpolation formula to the same two nodes. The
redshift must agree with the default to one rounding everywhere, including
at the nodes, beyond the table and below the first node, and the gradients
must be the same. The default must stay the historical arithmetic and stay
out of the run fingerprint.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.cosmology import distances as d
from darksirens.cosmology import configure_z_of_dL_lookup, z_of_dL_lookup
from darksirens.inference.run_fingerprint import (
    core_numerics_semantic,
    fingerprint_from_semantic,
)

N_NODES = d.zgrid.shape[0]


@pytest.fixture
def direct():
    configure_z_of_dL_lookup("direct")
    try:
        yield
    finally:
        configure_z_of_dL_lookup("search")


def _cosmologies(n_random=10, seed=3):
    """Table corners (extreme node spacing and span) plus random interior points."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(8):
        out.append((
            rng.uniform(20.0, 140.0),
            float(d.Om0grid[[0, -1][k % 2]]),
            float(d.w0grid[[0, -1][(k // 2) % 2]]),
            float(d.wagrid[[0, -1][(k // 4) % 2]]),
        ))
    for _ in range(n_random):
        out.append((
            rng.uniform(20.0, 140.0),
            rng.uniform(float(d.Om0grid[0]), float(d.Om0grid[-1])),
            rng.uniform(float(d.w0grid[0]), float(d.w0grid[-1])),
            rng.uniform(float(d.wagrid[0]), float(d.wagrid[-1])),
        ))
    return out


def _edge_distances(xp, rng):
    """Nodes and their float neighbours, cell edges and their neighbours,
    the first interval, beyond the table, zero, negatives and non-finite."""
    xp = np.asarray(xp)
    edges = xp[1] * np.exp(np.arange(d._DIRECT_CELLS + 1) * d._DIRECT_CELL_WIDTH)
    return np.concatenate([
        xp, np.nextafter(xp, np.inf), np.nextafter(xp, -np.inf),
        edges, np.nextafter(edges, np.inf), np.nextafter(edges, -np.inf),
        rng.uniform(0.0, xp[1], 500),
        rng.uniform(-10.0, 1.2 * xp[-1], 3000),
        [0.0, -0.0, -1.0, 1e-300, 5e-324, 1.5 * xp[-1], 1e30, np.inf, -np.inf, np.nan],
    ])


def _z(lookup, dL, xp):
    previous = z_of_dL_lookup()
    configure_z_of_dL_lookup(lookup)
    try:
        return d.z_of_dL_precomputed(dL, xp)
    finally:
        configure_z_of_dL_lookup(previous)


# --------------------------------------------------------------------------
# Default identity
# --------------------------------------------------------------------------

def test_default_is_the_historical_search():
    assert z_of_dL_lookup() == "search"
    assert d.Z_OF_DL_LOOKUPS == ("search", "direct")

    def historical(dL, dL_grid):
        in_grid = (dL >= dL_grid[0]) & (dL <= dL_grid[-1])
        z = (
            jnp.interp(dL, dL_grid, d.zgrid)
            if d._USE_INTERP_SCAN
            else d._interp_unrolled(dL, dL_grid, d.zgrid)
        )
        return jnp.where(in_grid, z, jnp.nan)

    historical.__name__ = historical.__qualname__ = "z_of_dL_precomputed"
    xp = d.dL_of_z(d.zgrid, 70.0)
    dL = jnp.linspace(0.0, 30000.0, 257)
    assert str(jax.make_jaxpr(d.z_of_dL_precomputed)(dL, xp)) == str(
        jax.make_jaxpr(historical)(dL, xp)
    )
    a = jax.jit(d.z_of_dL_precomputed).lower(dL, xp).compile().as_text()
    b = jax.jit(historical).lower(dL, xp).compile().as_text()
    strip = lambda text: "\n".join(  # noqa: E731
        line.split(", metadata=")[0] for line in text.splitlines()
    )
    assert strip(a) == strip(b)


def test_default_fingerprint_is_unchanged():
    numerics = core_numerics_semantic()
    assert numerics["distance_table_grid"] == {"zmax": d.zMax, "nodes": d._ZGRID_NODES}


# --------------------------------------------------------------------------
# Lookup geometry: two nodes never share a two-cell window, and the last
# node never reaches the last cell, for every row of the module table.
# --------------------------------------------------------------------------

def test_cell_geometry_holds_for_every_table_row():
    r = np.asarray(d.rs, dtype=np.float64).reshape(-1, N_NODES)
    one_plus_z = 1.0 + np.asarray(d.zgrid)
    u = np.log(one_plus_z[1:] * r[:, 1:]) - np.log(one_plus_z[1] * r[:, 1:2])
    cells = np.floor(u / d._DIRECT_CELL_WIDTH)
    assert np.all(np.diff(cells, axis=1) >= 2)
    assert np.max(cells) <= d._DIRECT_CELLS - 2
    # Any multiple of a row (H0) only shifts u; nodes spacing is H0-free.
    assert np.min(np.diff(u, axis=1)) > 2.0 * d._DIRECT_CELL_WIDTH


# --------------------------------------------------------------------------
# Opt-in accuracy, edges included
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cosmology", _cosmologies())
def test_direct_matches_search_at_the_edges(cosmology):
    rng = np.random.default_rng(int(1e6 * cosmology[0]) % 2**32)
    xp = d.dL_of_z(d.zgrid, *cosmology)
    dL = jnp.asarray(_edge_distances(xp, rng))

    # Eager evaluation: the same nodes and the same arithmetic, bit for bit.
    a = np.asarray(_z("search", dL, xp))
    b = np.asarray(_z("direct", dL, xp))
    np.testing.assert_array_equal(a, b)

    # Under jit XLA may contract the formula differently: one rounding.
    ja = np.asarray(jax.jit(lambda x, g: _z("search", x, g))(dL, xp))
    jb = np.asarray(jax.jit(lambda x, g: _z("direct", x, g))(dL, xp))
    np.testing.assert_array_equal(np.isnan(ja), np.isnan(jb))
    fin = np.isfinite(ja)
    assert np.all(np.abs(ja[fin] - jb[fin]) <= np.spacing(np.abs(ja[fin])))

    # Support: exactly the table's [dL_grid[0], dL_grid[-1]].
    xpn = np.asarray(xp)
    dLn = np.asarray(dL)
    # XLA:CPU flushes subnormals to zero, so -5e-324 counts as dL = 0.
    dLn = np.where(np.abs(dLn) < np.finfo(dLn.dtype).tiny, 0.0, dLn)
    inside = (dLn >= xpn[0]) & (dLn <= xpn[-1])
    np.testing.assert_array_equal(np.isfinite(b), inside)
    assert b[dLn == xpn[0]][0] == 0.0
    assert b[dLn == xpn[-1]][0] == a[dLn == xpn[-1]][0]


def test_direct_under_vmap_over_proposals():
    rng = np.random.default_rng(11)
    H0 = jnp.asarray(rng.uniform(20.0, 140.0, 12))
    Om0 = jnp.asarray(rng.uniform(0.2, 0.4, 12))
    dL = jnp.asarray(np.concatenate([rng.uniform(0.0, 40000.0, 4000), [0.0, 1e-300, np.nan]]))

    def z(lookup):
        def one(h0, om0):
            return _z(lookup, dL, d.dL_of_z(d.zgrid, h0, om0))
        return np.asarray(jax.jit(jax.vmap(one))(H0, Om0))

    zd, zs = z("direct"), z("search")
    np.testing.assert_array_equal(np.isnan(zd), np.isnan(zs))
    fin = np.isfinite(zs)
    assert np.all(np.abs(zd[fin] - zs[fin]) <= np.spacing(np.abs(zs[fin])))


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------

def test_direct_gradient_in_dL_takes_the_same_bracket():
    # The one-sided slope at a node identifies the bracket: equal derivatives
    # at every node and node neighbour mean the same interval was used.
    rng = np.random.default_rng(5)
    for cosmology in _cosmologies(n_random=3)[::2]:
        xp = d.dL_of_z(d.zgrid, *cosmology)
        dL = _edge_distances(xp, rng)
        dL = jnp.asarray(dL[np.isfinite(dL) & (dL >= 0.0) & (dL <= np.asarray(xp)[-1])])
        grads = {}
        for lookup in ("search", "direct"):
            fn = jax.vmap(jax.grad(lambda x, g, lk=lookup: _z(lk, x, g)), in_axes=(0, None))
            grads[lookup] = np.asarray(jax.jit(fn)(dL, xp))
        assert np.all(np.isfinite(grads["direct"]))
        np.testing.assert_array_equal(grads["direct"], grads["search"])


def test_direct_gradient_in_cosmology_is_finite_and_equal():
    rng = np.random.default_rng(9)
    dL = jnp.asarray(np.concatenate([
        rng.uniform(1.0, 30000.0, 5000),
        np.asarray(d.dL_of_z(d.zgrid, 70.0, 0.31))[:: 7],
        [0.0, 1e-300],
    ]))

    def loss(lookup):
        def f(H0, Om0, w0):
            configure_z_of_dL_lookup(lookup)
            z = d.z_of_dL_precomputed(dL, d.dL_of_z(d.zgrid, H0, Om0, w0))
            configure_z_of_dL_lookup("search")
            return jnp.sum(jnp.where(jnp.isfinite(z), jnp.log1p(z), 0.0))
        return jax.jit(jax.grad(f, argnums=(0, 1, 2)))

    for args in ((70.0, 0.31, -1.0), (35.0, 0.25, -0.8), (130.0, 0.4, -1.2)):
        gs, gd = loss("search")(*args), loss("direct")(*args)
        for a, b in zip(gs, gd):
            assert np.isfinite(float(b))
            np.testing.assert_allclose(float(b), float(a), rtol=1e-12, atol=0.0)


# --------------------------------------------------------------------------
# Setting and fingerprint
# --------------------------------------------------------------------------

def test_configure_validates_and_clears_the_jitted_inverse():
    with pytest.raises(ValueError, match="z_of_dL lookup"):
        configure_z_of_dL_lookup("bisect")
    assert configure_z_of_dL_lookup() == "search"
    d.z_of_dL(jnp.asarray([100.0, 1000.0]), 70.0)
    assert d.z_of_dL.jitted._cache_size() >= 1
    try:
        assert configure_z_of_dL_lookup("direct") == "direct"
        assert d.z_of_dL.jitted._cache_size() == 0
        z = d.z_of_dL(jnp.asarray([100.0, 1000.0]), 70.0)
        assert np.all(np.isfinite(np.asarray(z)))
    finally:
        configure_z_of_dL_lookup("search")
    assert z_of_dL_lookup() == "search"


def test_direct_enters_the_fingerprint_only_when_set(direct):
    numerics = core_numerics_semantic()
    assert numerics["distance_table_grid"]["z_of_dL_lookup"] == "direct"
    configure_z_of_dL_lookup("search")
    base = core_numerics_semantic()
    assert "z_of_dL_lookup" not in base["distance_table_grid"]
    assert (
        fingerprint_from_semantic({"core_numerics": numerics})["digest"]
        != fingerprint_from_semantic({"core_numerics": base})["digest"]
    )


_ENV_PROBE = r"""
import json
from darksirens.cosmology import z_of_dL_lookup
from darksirens.inference.run_fingerprint import core_numerics_semantic
print(json.dumps({"lookup": z_of_dL_lookup(),
                  "grid": core_numerics_semantic()["distance_table_grid"]}))
"""


def _env_probe(value):
    env = {k: v for k, v in os.environ.items() if not k.startswith("DARKSIRENS_")}
    env["JAX_PLATFORMS"] = "cpu"
    if value is not None:
        env["DARKSIRENS_Z_OF_DL_LOOKUP"] = value
    return subprocess.run(
        [sys.executable, "-c", _ENV_PROBE], capture_output=True, text=True, env=env, timeout=600
    )


def test_environment_variable_selects_the_lookup():
    proc = _env_probe("direct")
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["lookup"] == "direct"
    assert out["grid"]["z_of_dL_lookup"] == "direct"
    bad = _env_probe("bisect")
    assert bad.returncode != 0
    assert "DARKSIRENS_Z_OF_DL_LOOKUP" in bad.stderr


# --------------------------------------------------------------------------
# The float32 per-sample path (compute_dtype="float32") honours the lookup
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cosmology", _cosmologies(n_random=4)[::2])
def test_float32_path_takes_the_same_bracket(cosmology):
    from darksirens.likelihood import mixed_precision as mp

    rng = np.random.default_rng(17)
    xp32 = jnp.asarray(d.dL_of_z(d.zgrid, *cosmology)).astype(jnp.float32)
    dL = _edge_distances(np.asarray(xp32, dtype=np.float64), rng)
    dL32 = jnp.asarray(dL[np.isfinite(dL) | np.isnan(dL)].astype(np.float32))

    def z32(lookup):
        previous = z_of_dL_lookup()
        configure_z_of_dL_lookup(lookup)
        try:
            eager = np.asarray(mp.z_of_dL(dL32, xp32))
            jitted = np.asarray(jax.jit(mp.z_of_dL)(dL32, xp32))
        finally:
            configure_z_of_dL_lookup(previous)
        return eager, jitted

    (es, js), (ed, jd) = z32("search"), z32("direct")
    assert es.dtype == ed.dtype == np.float32
    np.testing.assert_array_equal(ed, es)
    np.testing.assert_array_equal(np.isnan(jd), np.isnan(js))
    fin = np.isfinite(js)
    assert np.all(np.abs(jd[fin] - js[fin]) <= np.spacing(np.abs(js[fin])))
