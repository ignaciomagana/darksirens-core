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


def log_interp_zgrid(z, log_grid):
    """Interpolate a log-density grid with the correct z^2 low-z limit."""
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
