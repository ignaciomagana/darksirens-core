"""Shared logarithmic redshift grid used by redshift-density models."""

from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

zMax: float = float(os.environ.get("DARKSIRENS_ZMAX", 5.0))
_ZGRID_NODES = max(1000, int(round(1000 * math.log(zMax + 1.0) / math.log(6.0))))
zgrid = jnp.expm1(jnp.linspace(jnp.log(1.0), jnp.log(zMax + 1.0), _ZGRID_NODES))
zgrid = zgrid.at[-1].set(
    jnp.minimum(zgrid[-1], float(np.expm1(np.log(zMax + 1.0))))
)

_DLOG = math.log(zMax + 1.0) / (_ZGRID_NODES - 1)
_INTERP_DX0_EPS = float(np.spacing(np.finfo(np.dtype(zgrid.dtype)).eps))
_USE_SEARCHSORTED = os.environ.get("DARKSIRENS_INTERP_SEARCHSORTED") == "1"

#: The low-redshift interpolation, read once at import from
#: ``DARKSIRENS_LOW_Z_INTERP``. ``"legacy"`` (the default) is the frozen
#: reference's. ``"corrected"`` is the opt-in low-z law of both redshift maps:
#: :func:`log_interp_zgrid` interpolates the smooth ``log_grid - 2 ln z`` and
#: adds ``2 ln z`` back (the legacy chord of a log density bending like
#: ``2 ln z`` is up to 0.12 low in log p near z = 0.0026 and 8e-3 low at
#: z = 0.01), and :func:`darksirens.cosmology.distances.z_of_dL_precomputed`
#: inverts the Hermite first distance cell exactly (the legacy chord there
#: puts z up to 0.6% low). The run fingerprint records a non-default value.
LOW_Z_INTERP_CHOICES = ("legacy", "corrected")
LOW_Z_INTERP = os.environ.get("DARKSIRENS_LOW_Z_INTERP", "legacy")
if LOW_Z_INTERP not in LOW_Z_INTERP_CHOICES:
    raise ValueError(
        f"DARKSIRENS_LOW_Z_INTERP={LOW_Z_INTERP!r}: expected one of "
        f"{LOW_Z_INTERP_CHOICES}"
    )


def zgrid_upper_index(z):
    """Upper bracketing node, equivalent to clipped searchsorted(side='right')."""
    x = jnp.asarray(z)
    x = x.astype(jnp.result_type(x.dtype, zgrid.dtype))
    n = zgrid.shape[0]
    if _USE_SEARCHSORTED:
        return jnp.clip(jnp.searchsorted(zgrid, x, side="right"), 1, n - 1)
    i = jnp.clip(
        jnp.floor(jnp.log1p(jnp.maximum(x, 0.0)) / _DLOG).astype(jnp.int32) + 1,
        1,
        n - 1,
    )
    i = i - ((i > 1) & (x < zgrid[i - 1]))
    i = i + ((i < n - 1) & (x >= zgrid[i]))
    return jnp.where(x >= zgrid[-1], n - 1, i)


def _interp_zgrid(z, log_grid):
    x = jnp.asarray(z)
    x = x.astype(jnp.result_type(x.dtype, zgrid.dtype))
    fp = jnp.asarray(log_grid)
    i = zgrid_upper_index(x)
    df = fp[i] - fp[i - 1]
    dx = zgrid[i] - zgrid[i - 1]
    delta = x - zgrid[i - 1]
    dx0 = jnp.abs(dx) <= _INTERP_DX0_EPS
    f = jnp.where(
        dx0,
        fp[i - 1],
        fp[i - 1] + (delta / jnp.where(dx0, 1, dx)) * df,
    )
    f = jnp.where(x < zgrid[0], fp[0], f)
    return jnp.where(x > zgrid[-1], fp[-1], f)


def _low_z_corrected_log_interp(z, log_grid, nodes, upper):
    """``log_grid`` at ``z`` as ``2 ln z`` plus the linearly interpolated residual.

    ``log_grid`` is a log density that bends like ``2 ln z`` at low z (a
    ``dV_c/dz`` measure), so its residual ``log_grid - 2 ln z`` is smooth and
    a chord of it is accurate where a chord of ``log_grid`` is not. ``upper``
    is the bracket :func:`zgrid_upper_index` returns on ``nodes`` (the grid in
    the evaluation dtype). Below ``nodes[1]`` the residual is extrapolated from
    the second cell, so the first cell keeps the ``z^2`` law times a smooth
    factor; above the grid the value is ``log_grid[-1]``, as in the legacy
    branch.
    """
    x = jnp.asarray(z)
    log_nodes = jnp.log(jnp.where(nodes > 0.0, nodes, nodes[1]))
    resid = jnp.asarray(log_grid) - 2.0 * log_nodes
    i = jnp.maximum(upper, 2)
    dx = nodes[i] - nodes[i - 1]
    dx0 = jnp.abs(dx) <= float(np.spacing(np.finfo(np.dtype(nodes.dtype)).eps))
    t = jnp.where(dx0, 0.0, (x - nodes[i - 1]) / jnp.where(dx0, 1, dx))
    value = resid[i - 1] + t * (resid[i] - resid[i - 1]) + 2.0 * jnp.log(
        jnp.maximum(x, jnp.finfo(nodes.dtype).tiny)
    )
    return jnp.where(x > nodes[-1], log_grid[-1], value)


def log_interp_zgrid(z, log_grid):
    """Interpolate a log-density grid with the correct z^2 low-z limit.

    Under ``DARKSIRENS_LOW_Z_INTERP=corrected`` (see :data:`LOW_Z_INTERP`)
    the z^2 law holds in every cell, not only below ``zgrid[1]``
    (:func:`_low_z_corrected_log_interp`).
    """
    if LOW_Z_INTERP == "corrected":
        return _low_z_corrected_log_interp(z, log_grid, zgrid, zgrid_upper_index(z))
    z1 = zgrid[1]
    below = log_grid[1] + 2.0 * jnp.log(
        jnp.maximum(z, jnp.finfo(zgrid.dtype).tiny) / z1
    )
    above = (
        jnp.interp(z, zgrid, log_grid)
        if _USE_SEARCHSORTED
        else _interp_zgrid(z, log_grid)
    )
    return jnp.where(z < z1, below, above)
