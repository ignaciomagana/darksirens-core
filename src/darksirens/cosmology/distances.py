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
import math
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

#: How :func:`z_of_dL_precomputed` finds the distance-table interval that
#: brackets each luminosity distance. ``"search"`` (default) is the
#: historical unrolled binary search over the per-proposal ``dL_grid``.
#: ``"direct"`` computes the interval from ``log(dL)`` with index arithmetic
#: on a per-proposal cell table (see :func:`_interp_direct`) and applies the
#: same linear interpolation to the same two nodes, so the redshift agrees
#: with the default to one rounding. ``"direct"`` takes precedence over
#: ``DARKSIRENS_INTERP_SCAN``.
Z_OF_DL_LOOKUPS = ("search", "direct")


def _validated_z_of_dL_lookup(value, source):
    if value not in Z_OF_DL_LOOKUPS:
        raise ValueError(
            f"z_of_dL lookup must be one of {Z_OF_DL_LOOKUPS}, got {value!r} ({source})"
        )
    return value


_Z_OF_DL_LOOKUP = _validated_z_of_dL_lookup(
    os.environ.get("DARKSIRENS_Z_OF_DL_LOOKUP", "search"),
    "env DARKSIRENS_Z_OF_DL_LOOKUP",
)


def z_of_dL_lookup() -> str:
    """The active :data:`Z_OF_DL_LOOKUPS` setting."""
    return _Z_OF_DL_LOOKUP


def configure_z_of_dL_lookup(lookup: str | None = None) -> str:
    """Select the z(dL) bracket lookup and return the active setting.

    ``lookup`` is ``"search"`` (default; env ``DARKSIRENS_Z_OF_DL_LOOKUP``)
    or ``"direct"``; ``None`` leaves the setting untouched. The setting is
    read when a likelihood is traced, so configure it before binding the
    analysis. Changing it clears the trace cache of this module's jitted
    :func:`z_of_dL`; a likelihood that was already jitted keeps the lookup
    it was traced with.
    """

    global _Z_OF_DL_LOOKUP
    if lookup is not None:
        lookup = _validated_z_of_dL_lookup(lookup, "configure_z_of_dL_lookup")
        if lookup != _Z_OF_DL_LOOKUP:
            _Z_OF_DL_LOOKUP = lookup
            z_of_dL.jitted.clear_cache()
    return _Z_OF_DL_LOOKUP


def _direct_lookup_geometry():
    """Static cell width and cell count of the ``"direct"`` z(dL) lookup.

    The lookup cuts ``u = log(dL / dL_grid[1])`` into cells of width ``h``.
    Consecutive table nodes ``k-1 < k`` (``k >= 2``) are separated in ``u``
    by ``log((1 + z_k) / (1 + z_{k-1})) + log(r_k / r_{k-1})``. The comoving
    distance never decreases along the grid, so the separation is at least
    the first term for any table and any H0 (H0 only shifts ``u``). With
    ``h`` below half of that, no window of two cells holds two nodes, so a
    cell index that rounding moved by one cell still puts the bracket within
    one node of the truth, and one comparison on each side fixes it.

    The cell count covers ``log(dL_grid[-1] / dL_grid[1])`` for every point
    of the module table: the multilinear interpolation in (Om0, w0, wa) is a
    nonnegative combination of table rows, so the ratio is bounded by its
    maximum over the rows.
    """

    log_one_plus_z = np.log1p(_zgrid_numpy)
    h = 0.45 * float(np.min(np.diff(log_one_plus_z)[1:]))
    table = np.asarray(rs, dtype=np.float64)
    span = float(log_one_plus_z[-1] - log_one_plus_z[1]) + float(
        np.max(np.log(table[..., -1]) - np.log(table[..., 1]))
    )
    return h, int(math.ceil(span / h)) + 2


_Z_FIRST_NODE = float(np.asarray(zgrid)[1])
_R_SLOPE_AT_ZERO = float(speed_of_light / H0Planck)


def _low_z_hermite_correction(r_lin, z):
    u = z / _Z_FIRST_NODE
    corrected = r_lin + (1.0 - u) * (_R_SLOPE_AT_ZERO * z - r_lin)
    return jnp.where((z > 0.0) & (z < _Z_FIRST_NODE), corrected, r_lin)


rs = jnp.asarray(_build_cpl_distance_grid())
_DIRECT_CELL_WIDTH, _DIRECT_CELLS = _direct_lookup_geometry()

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
    """Invert ``dL_grid = dL_of_z(zgrid, ...)``; NaN outside its support.

    The bracket lookup follows :func:`configure_z_of_dL_lookup`. The
    ``"direct"`` lookup relies on ``dL_grid`` being a distance-table row (a
    positive multiple of ``(1 + z) r(z)`` on ``zgrid``, strictly increasing).
    """
    dL_grid = jnp.asarray(dL_grid)
    n = dL_grid.shape[0]
    if dL_grid.ndim != 1 or n != zgrid.shape[0]:
        raise ValueError(
            "z_of_dL_precomputed: dL_grid must be a one-dimensional array of "
            f"{zgrid.shape[0]} nodes to match zgrid, got shape {dL_grid.shape}"
        )
    in_grid = (dL >= dL_grid[0]) & (dL <= dL_grid[-1])
    if _Z_OF_DL_LOOKUP == "direct":
        z = _interp_direct(dL, dL_grid, zgrid)
    elif _USE_INTERP_SCAN:
        z = jnp.interp(dL, dL_grid, zgrid)
    else:
        z = _interp_unrolled(dL, dL_grid, zgrid)
    return jnp.where(in_grid, z, jnp.nan)


def _direct_cells(log_value, log_first):
    """Cell of ``log_value`` in the ``"direct"`` lookup, before clipping."""
    u = (log_value - log_first) / _DIRECT_CELL_WIDTH
    return jnp.floor(jnp.clip(u, -1.0, float(_DIRECT_CELLS))).astype(jnp.int32)


def _direct_cell_table(xp, fp):
    """Per-proposal table of the ``"direct"`` z(dL) lookup.

    Cell ``j`` (width ``h`` in ``u = log(dL / xp[1])``) gets ``i0_j``, one
    plus the number of nodes ``k >= 1`` in cells below ``j``, clipped to
    ``[1, n - 1]``; for a sample in cell ``j`` this is within one node of
    the searched upper bracket (see :func:`_direct_lookup_geometry`). Row
    ``j`` stores ``xp`` and ``fp`` at nodes ``i0_j - 2 .. i0_j + 1`` (clamped
    to the table), which covers the bracket after a one-node correction, so
    a sample needs one row gather. ``i0_j`` comes from an unrolled search of
    the ``cells`` edges over the ``n - 1`` sorted node cells (once per
    proposal; no scatter, which XLA:CPU lowers to a serial loop). The cell
    assignment carries no gradient; the stored ``xp`` values do.
    """

    n = xp.shape[0]
    xs = jax.lax.stop_gradient(xp)
    log_first = jnp.log(xs[1])
    node_cells = _direct_cells(jnp.log(xs[1:]), log_first)
    i0 = 1 + jnp.searchsorted(
        node_cells,
        jnp.arange(_DIRECT_CELLS, dtype=node_cells.dtype),
        side="left",
        method="scan_unrolled",
    )
    i0 = jnp.clip(i0, 1, n - 1)
    idx = jnp.clip(i0[:, None] + jnp.arange(-2, 2)[None, :], 0, n - 1)
    return jnp.concatenate([xp[idx], fp[idx]], axis=1), log_first


def _interp_direct(x, xp, fp):
    """``_interp_unrolled`` with the bracket found by index arithmetic.

    The row of the sample's cell holds ``xp`` and ``fp`` at nodes ``i0 - 2 ..
    i0 + 1`` with ``i0`` within one node of the searched bracket ``i``.
    Comparing ``x`` with ``xp[i0 - 1]`` and ``xp[i0]`` recovers ``i`` exactly,
    including its clipping to ``[1, n - 1]`` (``i0 < n - 1`` is
    ``xp[i0] < xp[n - 1]`` and ``i0 > 1`` is ``xp[i0 - 1] > xp[0]`` on a
    strictly increasing table). The interpolation is the default's formula
    on the same two nodes; XLA may contract it differently, so ``z`` agrees
    with the default to the last bit or one rounding.
    """

    table, log_first = _direct_cell_table(xp, fp)
    xs = jax.lax.stop_gradient(x)
    cell = jnp.clip(
        _direct_cells(jnp.log(jnp.maximum(xs, jnp.finfo(xp.dtype).tiny)), log_first),
        0,
        _DIRECT_CELLS - 1,
    )
    row = table[cell]
    x_m2, x_m1, x_0, x_p1, f_m2, f_m1, f_0, f_p1 = (row[..., k] for k in range(8))
    up = (x >= x_0) & (x_0 < xp[-1])
    down = (x < x_m1) & (x_m1 > xp[0])
    x_lo = jnp.where(down, x_m2, jnp.where(up, x_0, x_m1))
    x_hi = jnp.where(down, x_m1, jnp.where(up, x_p1, x_0))
    f_lo = jnp.where(down, f_m2, jnp.where(up, f_0, f_m1))
    f_hi = jnp.where(down, f_m1, jnp.where(up, f_p1, f_0))
    df = f_hi - f_lo
    dx = x_hi - x_lo
    delta = x - x_lo
    dx0 = jnp.abs(dx) <= _interp_dx0_eps(x, xp)
    f = jnp.where(
        dx0,
        f_lo,
        f_lo + (delta / jnp.where(dx0, 1, dx)) * df,
    )
    f = jnp.where(x < xp[0], fp[0], f)
    f = jnp.where(x > xp[-1], fp[-1], f)
    return f


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
