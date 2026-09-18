"""JAX-compatible flat-CPL distances used by the siren likelihood.

The implementation is a parity port of the pinned legacy distance table. The
large table is threaded through JIT boundaries as an argument rather than
captured as an HLO literal.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from .. import _cosmology_support as _support
from ._interpolation import interpnd, interpnd_scalar_head
from .parameters import H0_FID, OM0_FID, W0_FID, WA_FID

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

zMax = float(os.environ.get("DARKSIRENS_ZMAX", 5))
H0Planck = np.float64(H0_FID)
Om0Planck = OM0_FID
w0Fiducial = W0_FID
waFiducial = WA_FID
speed_of_light = np.float64(299792.458)

# The axis geometry lives in darksirens._cosmology_support so the public
# Cosmology spec can reject an out-of-grid prior without importing jax.
_OM0_PRIOR_HALF_WIDTH = _support.OM0_PRIOR_HALF_WIDTH
_OM0_GRID_PAD = _support.OM0_GRID_PAD
_W0_PRIOR_HALF_WIDTH = _support.W0_PRIOR_HALF_WIDTH
_W0_GRID_PAD = _support.W0_GRID_PAD
_WA_PRIOR_HALF_WIDTH = _support.WA_PRIOR_HALF_WIDTH
_WA_GRID_PAD = _support.WA_GRID_PAD

Om0PriorLower = _support.OM0_PRIOR_LOWER
Om0PriorUpper = _support.OM0_PRIOR_UPPER
w0PriorLower = _support.W0_PRIOR_LOWER
w0PriorUpper = _support.W0_PRIOR_UPPER
waPriorLower = _support.WA_PRIOR_LOWER
waPriorUpper = _support.WA_PRIOR_UPPER

_ZGRID_NODES = max(500, int(round(500 * np.log(zMax + 1.0) / np.log(6.0))))
_zgrid_numpy = np.expm1(np.linspace(np.log(1), np.log(zMax + 1), _ZGRID_NODES))

Om0grid = jnp.linspace(_support.OM0_GRID_LOWER, _support.OM0_GRID_UPPER, 21)
w0grid = jnp.linspace(_support.W0_GRID_LOWER, _support.W0_GRID_UPPER, 41)
wagrid = jnp.linspace(_support.WA_GRID_LOWER, _support.WA_GRID_UPPER, 31)
zgrid = jnp.array(_zgrid_numpy)


def _check_support_constants():
    """The light support constants must describe exactly this grid.

    They are what Cosmology validates a prior against, so a drift between the
    two would either reopen the silent-truncation hole or reject usable priors.
    """
    assert (_support.OM0_FID, _support.W0_FID, _support.WA_FID) == (
        OM0_FID,
        W0_FID,
        WA_FID,
    ), "_cosmology_support fiducials drifted from cosmology.parameters"
    for name, grid in (("Om0", Om0grid), ("w0", w0grid), ("wa", wagrid)):
        lower, upper = _support.GRID_SUPPORT[name]
        assert float(grid[0]) == lower and float(grid[-1]) == upper, (
            f"{name} grid endpoints drifted from _cosmology_support"
        )


_check_support_constants()


def _cpl_E_numpy(z, Om0, w0, wa):
    one_plus_z = 1.0 + z
    dark_energy = (
        (1.0 - Om0)
        * one_plus_z ** (3.0 * (1.0 + w0 + wa))
        * np.exp(-3.0 * wa * z / one_plus_z)
    )
    return np.sqrt(Om0 * one_plus_z**3 + dark_energy)


def _cumulative_trapezoid(y, x):
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate(([0.0], np.cumsum(increments)))


def _build_cpl_distance_grid():
    r_grid = np.empty(
        (len(Om0grid), len(w0grid), len(wagrid), len(zgrid)), dtype=np.float64
    )
    z = np.asarray(zgrid, dtype=np.float64)
    for i, Om0 in enumerate(np.asarray(Om0grid)):
        for j, w0 in enumerate(np.asarray(w0grid)):
            for k, wa in enumerate(np.asarray(wagrid)):
                inv_E = 1.0 / _cpl_E_numpy(z, float(Om0), float(w0), float(wa))
                r_grid[i, j, k, :] = (
                    speed_of_light / H0Planck
                ) * _cumulative_trapezoid(inv_E, z)
    return r_grid


def _interp_dx0_eps(*operands):
    dtype = jnp.result_type(jnp.result_type(*operands), float)
    return float(np.spacing(np.finfo(dtype).eps))


def _interp_unrolled(x, xp, fp):
    n = xp.shape[0]
    i = jnp.clip(
        jnp.searchsorted(xp, x, side="right", method="scan_unrolled"), 1, n - 1
    )
    df = fp[i] - fp[i - 1]
    dx = xp[i] - xp[i - 1]
    delta = x - xp[i - 1]
    dx0 = jnp.abs(dx) <= _interp_dx0_eps(x, xp)
    f = jnp.where(
        dx0,
        fp[i - 1],
        fp[i - 1] + (delta / jnp.where(dx0, 1, dx)) * df,
    )
    f = jnp.where(x < xp[0], fp[0], f)
    f = jnp.where(x > xp[-1], fp[-1], f)
    return f


_USE_INTERP_SCAN = os.environ.get("DARKSIRENS_INTERP_SCAN") == "1"
_Z_FIRST_NODE = float(np.asarray(zgrid)[1])
_R_SLOPE_AT_ZERO = float(speed_of_light / H0Planck)


def _low_z_hermite_correction(r_lin, z):
    u = z / _Z_FIRST_NODE
    corrected = r_lin + (1.0 - u) * (_R_SLOPE_AT_ZERO * z - r_lin)
    return jnp.where((z > 0.0) & (z < _Z_FIRST_NODE), corrected, r_lin)


rs = jnp.asarray(_build_cpl_distance_grid())

_ACTIVE_DISTANCE_TABLE = contextvars.ContextVar(
    "darksirens_active_distance_table", default=None
)
_AMBIENT_JIT_CHANNELS: list = []


def distance_table():
    active = _ACTIVE_DISTANCE_TABLE.get()
    return rs if active is None else active


def resolve_distance_table(table=None):
    return distance_table() if table is None else table


@contextlib.contextmanager
def bound_distance_table(table):
    if table is None:
        yield
        return
    token = _ACTIVE_DISTANCE_TABLE.set(table)
    try:
        yield
    finally:
        _ACTIVE_DISTANCE_TABLE.reset(token)


def register_ambient_jit_channel(resolve, bind):
    _AMBIENT_JIT_CHANNELS.append((resolve, bind))


def threads_distance_table(**jit_kwargs):
    """JIT a function with its distance table supplied as a true argument."""

    def decorate(fn):
        signature = inspect.signature(fn)
        if "distance_table" not in signature.parameters:
            raise TypeError(
                f"{fn.__qualname__} must declare a 'distance_table' parameter"
            )

        @functools.wraps(fn)
        def _bind_then_call(*args, _ambient_extras=(), **kwargs):
            table = signature.bind(*args, **kwargs).arguments.get("distance_table")
            with contextlib.ExitStack() as stack:
                stack.enter_context(bound_distance_table(table))
                for (_, bind), value in zip(_AMBIENT_JIT_CHANNELS, _ambient_extras):
                    stack.enter_context(bind(value))
                return fn(*args, **kwargs)

        jitted = jax.jit(_bind_then_call, **jit_kwargs)

        @functools.wraps(fn)
        def public(*args, distance_table=None, **kwargs):
            return jitted(
                *args,
                distance_table=resolve_distance_table(distance_table),
                _ambient_extras=tuple(res() for res, _ in _AMBIENT_JIT_CHANNELS),
                **kwargs,
            )

        public.jitted = jitted
        return public

    return decorate


@jit
def E(z, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial):
    """Dimensionless flat-CPL expansion rate H(z)/H0."""
    one_plus_z = 1.0 + z
    dark_energy = (
        (1.0 - Om0)
        * one_plus_z ** (3.0 * (1.0 + w0 + wa))
        * jnp.exp(-3.0 * wa * z / one_plus_z)
    )
    return jnp.sqrt(Om0 * one_plus_z**3 + dark_energy)


@threads_distance_table()
def r_of_z(
    z,
    H0,
    Om0=Om0Planck,
    w0=w0Fiducial,
    wa=waFiducial,
    distance_table=None,
):
    """Line-of-sight comoving distance in Mpc."""
    interpolate = (
        interpnd_scalar_head
        if jnp.ndim(Om0) == 0 and jnp.ndim(w0) == 0 and jnp.ndim(wa) == 0
        else interpnd
    )
    r_lin = interpolate(
        (Om0, w0, wa, z),
        (Om0grid, w0grid, wagrid, zgrid),
        distance_table,
        fill_value=jnp.nan,
    )
    return _low_z_hermite_correction(r_lin, z) * (H0Planck / H0)


@threads_distance_table()
def dL_of_z(
    z,
    H0,
    Om0=Om0Planck,
    w0=w0Fiducial,
    wa=waFiducial,
    distance_table=None,
):
    """Luminosity distance in Mpc."""
    return (1 + z) * r_of_z(z, H0, Om0, w0, wa)


@threads_distance_table()
def distance_modulus(
    z,
    H0,
    Om0=Om0Planck,
    w0=w0Fiducial,
    wa=waFiducial,
    distance_table=None,
):
    dL = dL_of_z(z, H0, Om0, w0, wa)
    return 5.0 * jnp.log10(jnp.maximum(dL, 1e-12) * 1.0e5)


@threads_distance_table()
def dL_grid_bounds(
    H0,
    Om0=Om0Planck,
    w0=w0Fiducial,
    wa=waFiducial,
    distance_table=None,
):
    dL_grid = dL_of_z(zgrid, H0, Om0, w0, wa)
    return dL_grid[0], dL_grid[-1]


@threads_distance_table()
def dL_in_z_grid(
    dL,
    H0,
    Om0=Om0Planck,
    w0=w0Fiducial,
    wa=waFiducial,
    distance_table=None,
):
    dL_min, dL_max = dL_grid_bounds(H0, Om0, w0, wa)
    return (dL >= dL_min) & (dL <= dL_max)


@threads_distance_table()
def z_of_dL(
    dL,
    H0,
    Om0=Om0Planck,
    w0=w0Fiducial,
    wa=waFiducial,
    distance_table=None,
):
    """Invert luminosity distance, returning NaN outside the table support."""
    dL_grid = dL_of_z(zgrid, H0, Om0, w0, wa)
    return z_of_dL_precomputed(dL, dL_grid)


def z_of_dL_precomputed(dL, dL_grid):
    dL_grid = jnp.asarray(dL_grid)
    n = dL_grid.shape[0]
    if dL_grid.ndim != 1 or n != zgrid.shape[0]:
        raise ValueError(
            "z_of_dL_precomputed: dL_grid must be a one-dimensional array of "
            f"{zgrid.shape[0]} nodes to match zgrid, got shape {dL_grid.shape}"
        )
    in_grid = (dL >= dL_grid[0]) & (dL <= dL_grid[-1])
    z = (
        jnp.interp(dL, dL_grid, zgrid)
        if _USE_INTERP_SCAN
        else _interp_unrolled(dL, dL_grid, zgrid)
    )
    return jnp.where(in_grid, z, jnp.nan)


@threads_distance_table()
def dV_of_z(
    z,
    H0,
    Om0=Om0Planck,
    w0=w0Fiducial,
    wa=waFiducial,
    distance_table=None,
):
    """Differential comoving volume per steradian in Mpc^3."""
    return speed_of_light * r_of_z(z, H0, Om0, w0, wa) ** 2 / (
        H0 * E(z, Om0, w0, wa)
    )


@jit
def ddL_of_z(
    z,
    dL,
    H0,
    Om0=Om0Planck,
    w0=w0Fiducial,
    wa=waFiducial,
):
    """Derivative d(dL)/dz."""
    return dL / (1 + z) + speed_of_light * (1 + z) / (
        H0 * E(z, Om0, w0, wa)
    )


def ddL_of_z_precomputed(z, ddL_grid):
    return jnp.interp(z, zgrid, ddL_grid)
