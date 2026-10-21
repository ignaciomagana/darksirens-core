"""Ordinary non-LSS catalog completeness and missing-host count budget.

This module reconstructs the frozen legacy per-row count-ratio estimator and
adds the pooled one (:func:`build_pooled_count_ratio_cache`).
Survey fitting, aggregate/stratified selection models, Q/latent fields, and
field/global normalizers are deliberately out of scope.

The observed and expected count densities are smoothed by the same truncated
Gaussian operator with fixed width ``SIGMA_SMOOTH = 0.05``.  For one compact
catalog row,

    C(z) = clip[dN_obs_s(z) / dN_exp_s(z), 0, 1]
    dN_miss(z) = [1 - C(z)] dN_exp(z)

and a finite catalog depth means completeness is exactly zero above that depth,
so ``dN_miss`` relaxes to the full ``dN_exp`` there.  The depth is a survey
completeness statement, not an analysis cutoff; ``N_miss`` always integrates
over the full modeled redshift grid.

Pooled (``ds.model(..., count_ratio="pooled")``, the default of ``ds.model``
since 2026-10-10): the pooled count ratio of
:func:`build_pooled_count_ratio_cache`, which smooths the ratio itself, pools
the catalog's rows before the clip and takes its kernel width as an argument;
``count_ratio="pooled_kernel"`` (opt-in) is the same estimator with each
galaxy counted through its redshift kernel instead of its point redshift.
The functions here dispatch on the cache they are given: no cache or an
:class:`ObservedDensityCache` is the legacy per-row estimator (what
``ds.model(..., count_ratio="row")`` binds), bit for bit as before; a
:class:`PooledCountRatioCache` is the pooled one. The defaults of these
low-level functions are therefore the legacy estimator; only ``ds.model``'s
default changed.
"""

from __future__ import annotations

import contextlib
import contextvars
import warnings
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.scipy.special import ndtr

from darksirens.cosmology._grid import zgrid
from darksirens.cosmology.distances import (
    register_ambient_jit_channel,
    threads_distance_table,
)
from darksirens.cosmology.parameters import CosmologyParameters

from .redshift import log_galaxy_measure_grid
from .types import CatalogParameters, GalaxyCatalog, row_blocks

jax.config.update("jax_enable_x64", True)

SIGMA_SMOOTH: float = 0.05
_ZMAX: float = float(np.asarray(zgrid)[-1])
_SQRT2PI: float = float(np.sqrt(2.0 * np.pi))


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature weights for a possibly non-uniform grid."""

    w = np.zeros_like(x)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


_TRAPW_NP = _trapezoid_weights(np.asarray(zgrid, dtype=np.float64))


def _truncated_kernel_mass_np(z0: np.ndarray, sigma: float) -> np.ndarray:
    """Mass of ``N(.; z0, sigma)`` inside the modeled redshift interval."""

    from scipy.special import ndtr as scipy_ndtr

    return scipy_ndtr((_ZMAX - z0) / sigma) - scipy_ndtr(-z0 / sigma)


def _build_smoothing_operator() -> jnp.ndarray:
    """Build the frozen matched-kernel expected-count smoothing operator."""

    z = np.asarray(zgrid, dtype=np.float64)
    mass = np.maximum(_truncated_kernel_mass_np(z, SIGMA_SMOOTH), 1.0e-300)
    pdf = np.exp(-0.5 * ((z[:, None] - z[None, :]) / SIGMA_SMOOTH) ** 2)
    pdf /= _SQRT2PI * SIGMA_SMOOTH
    return jnp.asarray((pdf / mass[None, :]) * _TRAPW_NP[None, :])


_S_EXP = _build_smoothing_operator()
_ACTIVE_SMOOTHING_OPERATOR = contextvars.ContextVar(
    "darksirens_active_catalog_smoothing_operator", default=None
)


def smoothing_operator():
    """Return the active smoothing operator, defaulting to the frozen grid."""

    active = _ACTIVE_SMOOTHING_OPERATOR.get()
    return _S_EXP if active is None else active


@contextlib.contextmanager
def bound_smoothing_operator(operator):
    """Bind ``operator`` across a nested JIT trace."""

    if operator is None:
        yield
        return
    token = _ACTIVE_SMOOTHING_OPERATOR.set(operator)
    try:
        yield
    finally:
        _ACTIVE_SMOOTHING_OPERATOR.reset(token)


# Match the frozen legacy anti-HLO-constant mechanism.  The existing distance
# JIT wrapper will pass this 1000x1000 table as a true argument at every nested
# boundary reached after this module has been imported; unused ambient arguments
# are eliminated by XLA.
register_ambient_jit_channel(smoothing_operator, bound_smoothing_operator)


class ObservedDensityCache(NamedTuple):
    """Theta-independent smoothed observed count density, one row per catalog row."""

    dN_obs_kde: Any


class CompletionState(NamedTuple):
    """Proposal-dependent grids shared by every ordinary catalog row."""

    log_g_grid: Any
    dN_exp: Any
    dN_exp_smooth: Any
    z_depth: Any


class CompletionCurves(NamedTuple):
    """Ordinary per-row completeness and missing-host count curves."""

    f: Any
    dN_miss: Any
    C_eff: Any
    N_miss: Any
    C: Any


class GatheredCompletionCurves(NamedTuple):
    """Missing-host density kept as row-by-redshift factors (opt-in).

    The ``missing_density="gather"`` form of :class:`CompletionCurves`
    (:mod:`darksirens.catalog.settings`).  Instead of the ``(N_rows, N_z)``
    grids, it keeps what they are built from and evaluates

        C(row, i)       = clip(observed[row, i] / z_factor[i], 0, 1)   (count ratio)
                        = row_fraction[row] * z_factor[i]              (row-fraction selection)
        dN_miss(row, i) = (1 - C(row, i)) dN_exp[i],  or dN_exp[i] where depth_mask[i] is False

    at one ``(row, i)`` with :func:`gathered_missing_density`, the arithmetic
    of the grid builders element by element.  Exactly one of ``observed``
    (the theta-independent observed-count cache) and ``row_fraction`` is set.
    ``N_miss`` is the per-row trapezoid of ``dN_miss`` over the redshift grid,
    as on the grid.  The incomplete-catalog prior states accept it wherever
    they accept :class:`CompletionCurves`; nothing else reads it.
    """

    observed: Any
    row_fraction: Any
    z_factor: Any
    dN_exp: Any
    depth_mask: Any
    N_miss: Any


def _observed_density_row(row, catalog: GalaxyCatalog):
    """Frozen per-row observed-count KDE used only by completeness.

    Raw galaxy counts are used.  Catalog host weights and per-galaxy redshift
    uncertainties do not enter this estimator; the fixed 0.05 kernel must match
    the expected-count smoothing operator exactly.

    The redshifts are read in the grid's float64 whatever the catalog stores:
    the ``1e-300`` floor on a kernel's mass is zero in float32, so a float32
    padding slot beyond the grid (zero mass, zero density) would give 0/0.
    """

    zs = catalog.zgals[row].astype(zgrid.dtype)
    real = jnp.arange(zs.shape[0]) < catalog.ngals[row]
    mass = ndtr((_ZMAX - zs) / SIGMA_SMOOTH) - ndtr(-zs / SIGMA_SMOOTH)
    mass = jnp.maximum(mass, 1.0e-300)
    pdf = jnp.exp(
        -0.5 * ((zgrid[:, None] - zs[None, :]) / SIGMA_SMOOTH) ** 2
    )
    pdf = pdf / (_SQRT2PI * SIGMA_SMOOTH)
    kernel = (pdf / mass[None, :]) * real[None, :].astype(pdf.dtype)
    return kernel.sum(axis=1)


_batched_observed_density = jit(
    vmap(_observed_density_row, in_axes=(0, None))
)

#: Real galaxies more than this many ``SIGMA_SMOOTH`` below ``z = 0`` are
#: refused by the per-row observed-density cache: beyond about 8.3 widths the
#: mass of their kernel inside the grid rounds to 0 in float64 and its floor
#: (1e-300) turns the kernel into about 1e279 (at 7 widths the mass is still
#: good to about 1e-4).
ROW_KDE_WIDTHS_BELOW_ZERO_MAX: float = 7.0
#: Effective galaxies per row under the 0.05 kernel at the catalog's median
#: redshift (the median over the rows holding galaxies) below which the
#: per-row count ratio warns that its clip at 1 biases the completeness low.
ROW_WARN_EFFECTIVE: float = 1000.0
#: Bytes of one batch's ``(rows, N_z, N_max)`` float64 kernel in
#: :func:`build_observed_density_cache` when no ``batch_size`` is given.
OBSERVED_DENSITY_BATCH_BYTES: int = 2**30
_OBSERVED_DENSITY_BATCH_MAX: int = 512


def _row_count_ratio_checks(catalog: GalaxyCatalog) -> None:
    """Host checks of a catalog for the per-row count ratio (``count_ratio="row"``).

    Refuses real galaxies more than ``ROW_KDE_WIDTHS_BELOW_ZERO_MAX`` kernel
    widths below ``z = 0`` (``ValueError``), and warns when the rows are too
    sparse for the clip of each row's ratio: the Kish effective number of a
    row's galaxies under the 0.05 kernel at the catalog's median redshift,
    the median over the rows holding galaxies, below ``ROW_WARN_EFFECTIVE``.
    A row's ratio is unbiased before the clip and its noise is about
    ``1 / sqrt(N)``, so the clip lowers its mean by about ``0.4 / sqrt(N)``
    where the catalog is complete, more at lower redshift where a row holds
    fewer galaxies, and more as the rows get smaller (higher nside): those
    missing hosts are made by the estimator.  Neither check changes a value.
    """

    z = np.asarray(catalog.zgals)
    ng = np.asarray(catalog.ngals)
    if z.shape[0] == 0 or z.shape[1] == 0:
        return
    cols = np.arange(z.shape[1])[None, :]
    floor = -ROW_KDE_WIDTHS_BELOW_ZERO_MAX * SIGMA_SMOOTH
    n_deep, lowest = 0, np.inf
    n_bins = int(np.ceil(_ZMAX / POOLED_BIN_WIDTH))
    counts = np.zeros(n_bins, dtype=np.int64)
    for rows in row_blocks(z.shape[0], z.shape[1]):
        zr = np.asarray(z[rows], dtype=np.float64)
        real = cols < ng[rows, None]
        deep = real & (zr < floor)
        if deep.any():
            n_deep += int(np.count_nonzero(deep))
            lowest = min(lowest, float(zr[deep].min()))
        zp = zr[real & (zr > 0.0) & (zr <= _ZMAX)]
        idx = np.minimum((zp / POOLED_BIN_WIDTH).astype(np.int64), n_bins - 1)
        counts += np.bincount(idx, minlength=n_bins)
    if n_deep:
        raise ValueError(
            f"count_ratio='row': {n_deep} real galaxies lie more than "
            f"{ROW_KDE_WIDTHS_BELOW_ZERO_MAX:g} widths of the per-row count ratio's "
            f"fixed {SIGMA_SMOOTH:g} kernel below z = 0 (z < {floor:g}; the lowest "
            f"is {lowest:.6g}): the kernel's mass inside the redshift grid rounds to "
            "0 in float64 there, and the row would read as complete far beyond z = 0. "
            "Remove them, or use count_ratio='pooled', which does not count "
            "galaxies at z <= 0"
        )
    if counts.sum() == 0:
        return
    median = (np.searchsorted(np.cumsum(counts), 0.5 * counts.sum()) + 0.5) * POOLED_BIN_WIDTH
    effective = []
    for rows in row_blocks(z.shape[0], z.shape[1]):
        zr = np.asarray(z[rows], dtype=np.float64)
        real = (cols < ng[rows, None]) & (ng[rows, None] > 0)
        g = np.where(real, np.exp(-0.5 * ((zr - median) / SIGMA_SMOOTH) ** 2), 0.0)
        s1 = g.sum(axis=1)
        s2 = (g * g).sum(axis=1)
        held = ng[rows] > 0
        effective.append(np.where(s2 > 0.0, s1 * s1 / np.where(s2 > 0.0, s2, 1.0), 0.0)[held])
    effective = np.concatenate(effective)
    if effective.size == 0:
        return
    typical = float(np.median(effective))
    if typical < ROW_WARN_EFFECTIVE:
        warnings.warn(
            f"count_ratio='row': the per-row count ratio's {SIGMA_SMOOTH:g} kernel holds "
            f"a median of {typical:.0f} effective galaxies per row at the catalog's "
            f"median redshift {median:.3f}, fewer at lower redshift: the clip at 1 of "
            "each row's noisy ratio biases its completeness low (by about "
            f"{0.4 / np.sqrt(max(typical, 1.0)):.2f} where the catalog is complete), "
            "which adds missing hosts that grow as the rows get smaller (higher "
            "nside). count_ratio='pooled' pools the rows before the clip",
            UserWarning,
            stacklevel=3,
        )


def build_observed_density_cache(
    catalog: GalaxyCatalog,
    *,
    batch_size: int | None = None,
) -> ObservedDensityCache:
    """Precompute the theta-independent observed count KDE row-for-row.

    The even-split/padded batching is numerically inert and mirrors the mature
    legacy cache builder while using compact row ids directly.  No dense global
    pixel lookup is constructed.  Each batch forms a ``(rows, N_z, N_max)``
    float64 kernel before its sum, so without a ``batch_size`` the rows per
    batch are sized from ``OBSERVED_DENSITY_BATCH_BYTES`` (at most 512, the
    legacy batch, and at least 1).

    The catalog is checked first (:func:`_row_count_ratio_checks`): real
    galaxies more than ``ROW_KDE_WIDTHS_BELOW_ZERO_MAX`` kernel widths below
    ``z = 0`` raise ``ValueError``, and rows too sparse for the clip warn.
    """

    n_rows = int(np.shape(catalog.zgals)[0])
    out = np.empty((n_rows, int(zgrid.size)), dtype=np.float64)
    if n_rows == 0:
        return ObservedDensityCache(jnp.asarray(out))

    _row_count_ratio_checks(catalog)
    if batch_size is None:
        per_row = int(zgrid.size) * max(1, int(np.shape(catalog.zgals)[1])) * 8
        batch_size = max(
            1, min(_OBSERVED_DENSITY_BATCH_MAX, OBSERVED_DENSITY_BATCH_BYTES // per_row)
        )
    requested = max(1, min(int(batch_size), n_rows))
    n_chunks = -(-n_rows // requested)
    block = -(-n_rows // n_chunks)
    for start in range(0, n_rows, block):
        rows = np.arange(start, min(start + block, n_rows), dtype=np.int32)
        take = int(rows.size)
        if take < block:
            rows = np.concatenate([rows, np.repeat(rows[-1:], block - take)])
        values = np.asarray(
            _batched_observed_density(jnp.asarray(rows, dtype=jnp.int32), catalog)
        )
        out[start : start + take] = values[:take]
    return ObservedDensityCache(jnp.asarray(out))


#: Redshift bin width of the pooled count ratio's galaxy histogram.
POOLED_BIN_WIDTH: float = 1.0e-4
#: Default kernel width of the pooled count ratio.
POOLED_WINDOW_DEFAULT: float = 0.02
#: Effective galaxies per kernel window at the catalog's median redshift below
#: which the pooled count ratio is refused, and below which it warns.
POOLED_MIN_EFFECTIVE: float = 100.0
POOLED_WARN_EFFECTIVE: float = 1000.0
#: Fraction of galaxy-free rows above which a pooled count ratio without a
#: ``row_fraction`` warns that they count as surveyed.
POOLED_EMPTY_ROW_WARN: float = 0.1
#: Redshift-error classes of the pooled count ratio with
#: ``redshift_errors=True`` (``count_ratio="pooled_kernel"``): log-spaced,
#: this many per decade above ``POOLED_KERNEL_WIDTH_FLOOR``, the floor of a
#: galaxy's redshift error there (the catalog kernel's width floor).
POOLED_KERNEL_CLASSES_PER_DECADE: int = 16
POOLED_KERNEL_WIDTH_FLOOR: float = 1.0e-4
#: Reach of a galaxy's redshift kernel in its widths: a galaxy farther than
#: this below ``z = 0`` or above ``z_top`` is not counted there.
POOLED_KERNEL_REACH: float = 8.0
#: The pooled count ratio at the galaxies' point redshifts warns where it is
#: more than this factor above the one that counts each galaxy through its
#: redshift kernel (``redshift_errors=True``).
POOLED_POINT_REDSHIFT_WARN: float = 1.5


@jax.tree_util.register_static
@dataclass(frozen=True)
class PooledCacheDepth:
    """The survey depth a pooled count-ratio cache was built for.

    A static pytree node (no leaves): inside ``jax.jit`` the depth is still a
    Python value, compared with the catalog parameters' (never traced)
    ``z_depth``.
    """

    z_depth: float | None


class PooledCountRatioCache(NamedTuple):
    """Theta-independent operands of the pooled count ratio (the ``ds.model`` default).

    ``operator`` is ``(N_z, N_z)``: applied to ``z^2 / dN_exp`` on the
    redshift grid it gives the pooled ratio ``R(z)`` of
    :func:`build_pooled_count_ratio_cache`.  ``row_fraction`` is the coverage
    fraction of each row of the catalog view the cache is used with.  It
    takes the place of :class:`ObservedDensityCache` wherever the count-ratio
    curves are evaluated.  ``z_top`` (0-d) is the depth the operator was
    normalised to (``z_depth``, or the top of the grid without one); the
    curves are NaN when it is not the evaluated ``params.z_depth`` (the top
    of the grid for ``None``), as a failed kernel-pin probe poisons the
    likelihood.  ``depth`` is the :class:`PooledCacheDepth` the builder
    records (its ``z_depth``, a static node): the completion curves refuse
    the cache (``ValueError``) at another ``CatalogParameters.z_depth``
    wherever that can be decided when the curves are traced, and ``z_top``
    catches a traced depth of another value.  A cache built without them
    (``None``, assembled by hand) is not checked.
    """

    operator: Any
    row_fraction: Any
    z_top: Any = None
    depth: Any = None


def _require_cache_depth(cache: PooledCountRatioCache, params: CatalogParameters) -> None:
    """Refuse a pooled cache built for another survey depth than ``params``'s.

    Inside the bound likelihood ``params.z_depth`` is a Python value; the
    public jitted entry points trace it, and then only its presence (a depth
    or ``None``) is compared.
    """

    depth = getattr(cache, "depth", None)
    if depth is None:
        return
    z_depth = params.z_depth
    if (z_depth is None) != (depth.z_depth is None):
        same = False
    elif z_depth is None or isinstance(z_depth, jax.core.Tracer):
        same = True
    else:
        same = float(z_depth) == depth.z_depth
    if not same:
        if z_depth is None:
            shown = "None"
        elif isinstance(z_depth, jax.core.Tracer):
            shown = "a (traced) depth"
        else:
            shown = repr(float(z_depth))
        raise ValueError(
            f"the pooled count-ratio cache was built for z_depth={depth.z_depth!r}, "
            f"but the catalog parameters have z_depth={shown}: its "
            "operator is normalised by the kernel mass inside the depth it was built "
            "for. Build it with build_pooled_count_ratio_cache(..., "
            "z_depth=<the catalog's z_depth>)"
        )


def _pooled_histogram(catalog: GalaxyCatalog, z_top: float):
    """Counts of the real galaxies with ``0 < z <= z_top`` in bins of ``POOLED_BIN_WIDTH``."""

    z = np.asarray(catalog.zgals)
    ng = np.asarray(catalog.ngals)
    n_bins = max(1, int(np.ceil(z_top / POOLED_BIN_WIDTH)))
    counts = np.zeros(n_bins, dtype=np.int64)
    cols = np.arange(z.shape[1])[None, :]
    n_low = 0
    # Row blocks: one block's redshifts in float64 at a time, never the catalog's.
    for rows in row_blocks(z.shape[0], z.shape[1]):
        zr = np.asarray(z[rows], dtype=np.float64)[cols < ng[rows, None]]
        n_low += int(np.count_nonzero(zr <= 0.0))
        zr = zr[(zr > 0.0) & (zr <= z_top)]
        idx = np.minimum((zr / POOLED_BIN_WIDTH).astype(np.int64), n_bins - 1)
        counts += np.bincount(idx, minlength=n_bins)
    return counts, n_low


def _positive_moments_np(z: np.ndarray, s: np.ndarray):
    """``m2`` and ``m3 / m2``, with ``m_k`` the integral of ``t^k N(t; z, s)`` over ``t > 0``.

    With ``x = z / s``: ``m2 = s^2 [(x^2 + 1) Phi(x) + x phi(x)]`` and
    ``m3 = s^3 [(x^3 + 3x) Phi(x) + (x^2 + 2) phi(x)]``.  Below ``x = 0`` both
    are written with the Mills ratio ``Phi(x) / phi(x)`` (``erfcx``), which
    keeps the cancellation of the tail at the level of rounding.  ``m3 / m2``
    is the mean of ``t`` under ``t^2 N(t; z, s)`` and tends to ``z`` as
    ``s -> 0``, as ``m2`` tends to ``z^2``.
    """

    from scipy.special import erfcx
    from scipy.special import ndtr as scipy_ndtr

    x = z / s
    phi = np.exp(-0.5 * x * x) / _SQRT2PI
    xn = np.minimum(x, 0.0)
    mills = np.sqrt(0.5 * np.pi) * erfcx(-xn / np.sqrt(2.0))
    pos = x >= 0.0
    cdf = scipy_ndtr(x)
    a2 = np.where(pos, (x * x + 1.0) * cdf + x * phi, phi * ((xn * xn + 1.0) * mills + xn))
    a3 = np.where(
        pos,
        (x**3 + 3.0 * x) * cdf + (x * x + 2.0) * phi,
        phi * ((xn**3 + 3.0 * xn) * mills + xn * xn + 2.0),
    )
    return s * s * a2, s * a3 / a2


def _pooled_kernel_cells(catalog: GalaxyCatalog, z_top: float):
    """The real galaxies as cells of (redshift bin, redshift-error class).

    Cells are ``POOLED_BIN_WIDTH`` wide in redshift and
    ``1 / POOLED_KERNEL_CLASSES_PER_DECADE`` decades wide in the error
    ``s_i = max(dz_i, POOLED_KERNEL_WIDTH_FLOOR)``.  Returns each cell's bin
    centre, rms error, galaxy count, weight ``sum_i 1 / m2(z_i, s_i)`` and
    weight-mean ``m3 / m2`` (:func:`_positive_moments_np`), and the number of
    real galaxies farther than ``POOLED_KERNEL_REACH`` widths below ``z = 0``.
    Those, and the galaxies as far above ``z_top``, are not counted.
    """

    z = np.asarray(catalog.zgals)
    dz = np.asarray(catalog.dzgals)
    ng = np.asarray(catalog.ngals)
    cols = np.arange(z.shape[1])[None, :]
    keys, number, weight, mean, square = [], [], [], [], []
    n_out = 0
    for rows in row_blocks(z.shape[0], z.shape[1]):
        real = cols < ng[rows, None]
        zr = np.asarray(z[rows], dtype=np.float64)[real]
        sr = np.maximum(np.asarray(dz[rows], dtype=np.float64)[real], POOLED_KERNEL_WIDTH_FLOOR)
        below = zr <= -POOLED_KERNEL_REACH * sr
        n_out += int(np.count_nonzero(below))
        inside = ~below & (zr < z_top + POOLED_KERNEL_REACH * sr)
        zr, sr = zr[inside], sr[inside]
        if zr.size == 0:
            continue
        m2, zhat = _positive_moments_np(zr, sr)
        w = 1.0 / m2
        zbin = np.floor(zr / POOLED_BIN_WIDTH).astype(np.int64)
        sclass = np.floor(
            np.log10(sr / POOLED_KERNEL_WIDTH_FLOOR) * POOLED_KERNEL_CLASSES_PER_DECADE
        ).astype(np.int64)
        block, inverse = np.unique(zbin * 4096 + sclass, return_inverse=True)
        keys.append(block)
        number.append(np.bincount(inverse))
        weight.append(np.bincount(inverse, weights=w))
        mean.append(np.bincount(inverse, weights=w * zhat))
        square.append(np.bincount(inverse, weights=w * sr * sr))
    if not keys:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty, empty, empty, n_out
    cell, inverse = np.unique(np.concatenate(keys), return_inverse=True)
    total = np.bincount(inverse, weights=np.concatenate(weight))
    centres = (np.floor_divide(cell, 4096) + 0.5) * POOLED_BIN_WIDTH
    widths = np.sqrt(np.bincount(inverse, weights=np.concatenate(square)) / total)
    zhat = np.bincount(inverse, weights=np.concatenate(mean)) / total
    count = np.bincount(inverse, weights=np.concatenate(number).astype(np.float64))
    return centres, widths, count, total, zhat, n_out


def _pooled_kernel_columns(z: np.ndarray, centres, widths, window: float, z_top: float):
    """``int_0^z_top G_window(z_k - t) N(t; c, s) dt`` at the grid nodes, one column per cell."""

    from scipy.special import ndtr as scipy_ndtr

    var = window * window + widths * widths
    total = np.sqrt(var)
    mu = (z[:, None] * (widths * widths)[None, :] + (centres * window * window)[None, :]) / var
    sd = (window * widths / total)[None, :]
    mass = scipy_ndtr((z_top - mu) / sd) - scipy_ndtr(-mu / sd)
    pdf = np.exp(-0.5 * ((z[:, None] - centres[None, :]) / total[None, :]) ** 2)
    return pdf / (_SQRT2PI * total[None, :]) * mass


def _point_redshift_excess(
    z: np.ndarray, centres, number, kernel, window: float, z_top: float
):
    """Where counting galaxies at their point redshifts moves the pooled ratio.

    ``kernel`` is :func:`_pooled_kernel_cells` of the same catalog.  Both
    ratios are evaluated with ``z^2 / dN_exp`` held at 1 (the low-redshift
    form of the expected density, which a proposal only rescales there), at
    the grid nodes in ``(0, z_top]``.  Returns ``(max ratio, node of the
    max, highest node where the ratio exceeds POOLED_POINT_REDSHIFT_WARN)``,
    or ``None`` where it never does.
    """

    k_centres, k_widths, _, k_weight, _, _ = kernel
    nodes = np.flatnonzero((z > 0.0) & (z <= z_top))
    if nodes.size == 0 or k_centres.size == 0:
        return None
    zk = z[nodes]
    point = np.zeros(zk.size)
    smooth = np.zeros(zk.size)
    weights = number / centres**2
    for start in range(0, centres.size, 4096):
        part = slice(start, start + 4096)
        pdf = np.exp(-0.5 * ((zk[:, None] - centres[None, part]) / window) ** 2)
        point += pdf @ weights[part] / (_SQRT2PI * window)
    for start in range(0, k_centres.size, 4096):
        part = slice(start, start + 4096)
        columns = _pooled_kernel_columns(zk, k_centres[part], k_widths[part], window, z_top)
        smooth += columns @ k_weight[part]
    ratio = np.where(smooth > 0.0, point / np.where(smooth > 0.0, smooth, 1.0), 1.0)
    high = np.flatnonzero(ratio > POOLED_POINT_REDSHIFT_WARN)
    if high.size == 0:
        return None
    worst = int(np.argmax(ratio))
    return float(ratio[worst]), float(zk[worst]), float(zk[high[-1]])


def build_pooled_count_ratio_cache(
    catalog: GalaxyCatalog,
    *,
    window: float = POOLED_WINDOW_DEFAULT,
    z_depth: float | None = None,
    row_fraction=None,
    redshift_errors: bool = False,
) -> PooledCountRatioCache:
    """Precompute the pooled count ratio of a catalog (``count_ratio="pooled"``, the default).

    The default estimator smooths the observed and the expected density and
    divides.  Both rise steeply with redshift, so that ratio at ``z`` is, to
    first order in the squared kernel width ``w^2``,
    ``C + w^2 (C' dlnE/dz + C''/2)`` with ``E = dN_exp/dz``: the completeness
    near ``z + 2 w^2 / z``, too low wherever the completeness falls.  This
    one smooths the ratio itself,

        R(z) = sum_i G_w(z - z_i) / dN_exp(z_i) / [F kappa(z)],

    over every real galaxy ``i`` of ``catalog`` (all its rows), with ``G_w``
    the Gaussian of width ``window``, ``F = sum_p f_p`` the summed coverage
    of the rows and ``kappa`` the kernel mass inside ``(0, z_top]``
    (``z_top`` is ``z_depth``, or the end of the redshift grid without one).
    Its expectation is the kernel mean of the completeness over
    ``(0, z_top]``: the ``C' dlnE/dz`` term is gone, the ``w^2 C''/2`` of any
    kernel remains, so the width should be small.  Every row then has

        C_p(z) = f_p clip[R(z), 0, 1],

    so the rows are pooled before the clip: the catalog is ONE completeness
    cell, the rows that ``row_fraction`` covers (every row without one).  A
    catalog whose depth varies over the sky is not one such cell, and
    splitting it into one catalog per depth under the field weighting is not
    the model of one catalog with a completeness per region: in every row of
    one region the other catalogs hold no galaxy and count as wholly missing
    hosts, so each region's galaxy contrast is diluted by its mixture weight.

    ``dN_exp`` depends on the proposal, so what is stored is linear in it:
    ``1 / dN_exp(z_i)`` is ``z_i^-2`` times the linear interpolation of
    ``z^2 / dN_exp`` between the grid nodes, and the galaxies enter through
    their counts in redshift bins of ``POOLED_BIN_WIDTH``.  Real galaxies
    with ``z <= 0`` or ``z > z_top`` are not counted.  The catalog is read in
    row blocks; the cache is one ``(N_z, N_z)`` operator and one value per
    row, in place of the ``(N_rows, N_z)`` observed-density cache.  The
    weight ``1/dN_exp(z_i)`` grows as ``z_i^-2``: within a few windows of
    ``z = 0`` a galaxy scattered there by its redshift error dominates the
    sum, the ratio exceeds 1 and the clip makes the catalog complete there,
    whatever its completeness, so the missing hosts there are removed.  A
    ``UserWarning`` names the redshifts where the ratio exceeds the one of
    ``redshift_errors=True`` (below) by more than the factor
    ``POOLED_POINT_REDSHIFT_WARN`` (photometric redshifts; galaxies at a
    redshift below their peculiar-velocity error).

    ``redshift_errors=True`` (``count_ratio="pooled_kernel"``, opt-in) counts
    each galaxy through its redshift kernel instead of its point redshift:
    the pooled ratio of the catalog's own host density,

        R(z) = sum_i int_0^z_top G_w(z - t) p_i(t) / dN_exp(t) dt / [F kappa(z)],

    with ``p_i(t) = N(t; z_i, s_i) dN_exp(t) / int N(t'; z_i, s_i) dN_exp(t') dt'``
    the galaxy's kernel and ``s_i = max(dz_i, POOLED_KERNEL_WIDTH_FLOOR)``
    its redshift error (``sigma_kde`` is not included: it may be sampled).
    The normaliser is taken at the low-redshift form ``dN_exp ~ t^2 /
    u(zhat_i)``, ``u = z^2 / dN_exp`` read at the mean ``zhat_i`` of ``t``
    under ``t^2 N(t; z_i, s_i)``, so the cache stays linear in ``u``: each
    galaxy weighs ``u(zhat_i) / m2_i`` (:func:`_positive_moments_np`), its
    kernel ``int_0^z_top G_w(z - t) N(t; z_i, s_i) dt`` is a Gaussian of width
    ``sqrt(w^2 + s_i^2)`` times a truncation, and galaxies at ``z <= 0`` or
    above ``z_top`` count through the part of their kernel inside.  Its
    expectation is the kernel mean of what the catalog term holds, so the
    observed and the missing hosts add up to ``dN_exp`` for a catalog that
    carries no host information, with or without redshift errors.  Galaxies
    are binned in redshift (``POOLED_BIN_WIDTH``) and in error classes
    (``POOLED_KERNEL_CLASSES_PER_DECADE``); one farther than
    ``POOLED_KERNEL_REACH`` widths above ``z_top`` is not counted, nor one as
    far below 0 (a ``UserWarning``).  For errors far below the window it is
    the default estimator.

    ``row_fraction`` values must be finite and in [0, 1], and a row with
    fraction 0 (outside the footprint) must hold no galaxy: its galaxies
    would be counted in the ratio of every row, and in its own row as hosts
    on top of the full missing-host density.  Both raise ``ValueError``.

    The cache records ``z_depth`` (its ``depth`` and ``z_top``,
    :class:`PooledCountRatioCache`), and the completion curves refuse it, or
    are NaN, at another ``CatalogParameters.z_depth``: pass the depth of the
    catalog the cache completes, ``None`` only for a catalog without one.

    A ``ValueError`` is raised when the kernel holds fewer than
    ``POOLED_MIN_EFFECTIVE`` effective galaxies at the catalog's median
    redshift, and a ``UserWarning`` below ``POOLED_WARN_EFFECTIVE``.
    """

    window = float(window)
    if not np.isfinite(window) or window <= 0.0:
        raise ValueError(f"the count-ratio window must be finite and > 0, got {window!r}")
    z = np.asarray(zgrid, dtype=np.float64)
    z_top = _ZMAX if z_depth is None else float(z_depth)
    if not 0.0 < z_top <= _ZMAX:
        raise ValueError(f"z_depth must lie in (0, {_ZMAX}], got {z_depth!r}")
    n_rows = int(np.shape(catalog.zgals)[0])
    ngals = np.asarray(catalog.ngals)
    name = "pooled_kernel" if redshift_errors else "pooled"
    if row_fraction is None:
        fraction = np.ones(n_rows, dtype=np.float64)
        n_empty = int(np.count_nonzero(ngals == 0))
        if n_empty > POOLED_EMPTY_ROW_WARN * n_rows:
            warnings.warn(
                f"count_ratio={name!r}: {n_empty} of {n_rows} catalog rows hold no "
                "galaxy and count as fully surveyed, which lowers the pooled "
                "completeness of every row; pass row_fraction (0 outside the "
                "footprint) if they are not surveyed",
                UserWarning,
                stacklevel=2,
            )
    else:
        fraction = np.asarray(row_fraction, dtype=np.float64)
        if fraction.shape != (n_rows,):
            raise ValueError(
                "row_fraction must contain exactly one value per catalog row; "
                f"got {fraction.shape} for {n_rows} rows"
            )
        if not np.all(np.isfinite(fraction)) or np.any(fraction < 0.0) or np.any(
            fraction > 1.0
        ):
            raise ValueError("row_fraction values must be finite and lie in [0, 1]")
    total = float(fraction.sum())
    if not total > 0.0:
        raise ValueError(f"count_ratio={name!r} needs a positive summed row coverage")
    if row_fraction is not None:
        refuse_galaxies_outside_row_fraction(fraction, ngals)

    if redshift_errors:
        centres, widths, number, weights, nodes, n_out = _pooled_kernel_cells(
            catalog, z_top
        )
        if n_out:
            warnings.warn(
                f"count_ratio='pooled_kernel': {n_out} real galaxies more than "
                f"{POOLED_KERNEL_REACH:g} redshift errors below z = 0 are not counted",
                UserWarning,
                stacklevel=2,
            )
        if centres.size == 0:
            raise ValueError(
                "count_ratio='pooled_kernel': the catalog holds no galaxy whose "
                f"redshift kernel reaches (0, {z_top}]"
            )
    else:
        counts, n_low = _pooled_histogram(catalog, z_top)
        if n_low:
            warnings.warn(
                f"count_ratio='pooled': {n_low} real galaxies at z <= 0 are not counted",
                UserWarning,
                stacklevel=2,
            )
        filled = np.flatnonzero(counts)
        if filled.size == 0:
            raise ValueError(
                f"count_ratio='pooled': the catalog holds no galaxy in (0, {z_top}]"
            )
        centres = (filled + 0.5) * POOLED_BIN_WIDTH
        number = counts[filled].astype(np.float64)
        weights = number / centres**2
        nodes = centres

    # Noise of the pooled ratio where the catalog is best sampled: the Kish
    # effective number of galaxies under one kernel at the median redshift
    # (each galaxy weighed by its cell's mean 1 / m2 with redshift errors).
    median = centres[np.searchsorted(np.cumsum(number), 0.5 * number.sum())]
    kernel = np.exp(-0.5 * ((median - centres) / window) ** 2)
    kernel = kernel * (weights / number) if redshift_errors else kernel / centres**2
    effective = float(np.sum(number * kernel) ** 2 / np.sum(number * kernel**2))
    text = (
        f"count_ratio={name!r}: a kernel of width {window:g} holds {effective:.0f} "
        f"effective galaxies at the catalog's median redshift {median:.3f} "
        f"({int(number.sum())} galaxies pooled), a relative noise of about "
        f"{1.0 / np.sqrt(effective):.2f} in the completeness"
    )
    if effective < POOLED_MIN_EFFECTIVE:
        raise ValueError(
            f"{text}; fewer than {POOLED_MIN_EFFECTIVE:g} is refused. Use a wider "
            "count_ratio_window, pool more rows, or completeness='selection'; "
            "count_ratio='row' runs the legacy per-row estimator (the default "
            "before 2026-10-10, biased where the completeness falls)"
        )
    if effective < POOLED_WARN_EFFECTIVE:
        warnings.warn(
            f"{text}; below {POOLED_WARN_EFFECTIVE:g} the clip at 1 biases the "
            "completeness low. Consider a wider count_ratio_window",
            UserWarning,
            stacklevel=2,
        )
    if not redshift_errors:
        excess = _point_redshift_excess(
            z, centres, number, _pooled_kernel_cells(catalog, z_top), window, z_top
        )
        if excess is not None:
            worst, at, below = excess
            warnings.warn(
                "count_ratio='pooled' counts each galaxy at its point redshift, with "
                "the weight 1/dN_exp that grows as z^-2: with this catalog's redshift "
                f"errors that ratio is up to {worst:.3g} times the one of the catalog's "
                f"own host density (at z = {at:.3g}), and more than "
                f"{POOLED_POINT_REDSHIFT_WARN:g} times it up to z = {below:.3g}; there "
                "the clip makes the catalog complete and removes its missing hosts "
                "whatever its completeness. count_ratio='pooled_kernel' counts each "
                "galaxy through its redshift kernel",
                UserWarning,
                stacklevel=2,
            )

    # A[k, j] = sum_b G_w(z_k - c_b) (n_b / c_b^2) phi_j(c_b), phi_j the hat
    # function of node j: the galaxy sum with 1/dN_exp left as a vector. With
    # redshift errors the cell's truncated kernel replaces G_w(z_k - c_b), its
    # weight sum_i 1/m2_i replaces n_b / c_b^2, and u is read at its zhat.
    nodes = np.clip(nodes, 0.0, z[-1])
    upper = np.clip(np.searchsorted(z, nodes, side="right"), 1, z.size - 1)
    t = (nodes - z[upper - 1]) / (z[upper] - z[upper - 1])
    operator = np.zeros((z.size, z.size), dtype=np.float64)
    for start in range(0, centres.size, 4096):
        part = slice(start, start + 4096)
        if redshift_errors:
            pdf = _pooled_kernel_columns(z, centres[part], widths[part], window, z_top)
            pdf *= weights[None, part]
        else:
            pdf = np.exp(-0.5 * ((z[:, None] - centres[None, part]) / window) ** 2)
            pdf *= weights[None, part] / (_SQRT2PI * window)
        hat = np.zeros((pdf.shape[1], z.size), dtype=np.float64)
        bins = np.arange(pdf.shape[1])
        hat[bins, upper[part] - 1] = 1.0 - t[part]
        hat[bins, upper[part]] = t[part]
        operator += pdf @ hat

    from scipy.special import ndtr as scipy_ndtr

    kappa = scipy_ndtr((z_top - z) / window) - scipy_ndtr(-z / window)
    inside = z <= z_top
    operator = np.where(
        inside[:, None],
        operator / (total * np.where(inside, kappa, 1.0))[:, None],
        0.0,
    )
    return PooledCountRatioCache(
        operator=jnp.asarray(operator),
        row_fraction=jnp.asarray(fraction),
        z_top=jnp.asarray(z_top, dtype=jnp.float64),
        depth=PooledCacheDepth(None if z_depth is None else float(z_depth)),
    )


def galaxies_outside_row_fraction(row_fraction, ngals) -> str | None:
    """Describe the real galaxies in rows whose coverage fraction is 0 (``None``: none).

    A row with ``row_fraction`` 0 lies outside the footprint, so the
    completeness there is 0 and the row's missing-host density is the full
    ``dN_exp``.  Galaxies stored in such a row (a coarser or misaligned
    mask) are counted as hosts on top of it, and under the pooled count
    ratio also in the ratio of every row while adding nothing to the summed
    coverage.
    """

    fraction = np.asarray(row_fraction, dtype=np.float64).reshape(-1)
    held = np.asarray(ngals).reshape(-1)
    rows = np.flatnonzero((fraction == 0.0) & (held > 0))
    if not rows.size:
        return None
    shown = ", ".join(str(int(r)) for r in rows[:8]) + (", ..." if rows.size > 8 else "")
    return (
        f"row_fraction is 0 in {rows.size} catalog rows that hold "
        f"{int(held[rows].sum())} real galaxies (rows {shown}): a row outside the "
        "footprint holds no galaxy, and these are counted as hosts on top of the "
        "row's full missing-host density"
    )


def refuse_galaxies_outside_row_fraction(row_fraction, ngals) -> None:
    """Raise ``ValueError`` for real galaxies in rows whose coverage fraction is 0.

    The pooled count ratio refuses them (:func:`galaxies_outside_row_fraction`).
    """

    text = galaxies_outside_row_fraction(row_fraction, ngals)
    if text is not None:
        raise ValueError(
            f"{text}. Give those rows their coverage fraction, or remove their galaxies"
        )


def _pooled_depth_ok(params: CatalogParameters, cache: PooledCountRatioCache):
    """Whether the cache was normalised to the depth it is evaluated at (``None`` unchecked)."""

    if cache.z_top is None:
        return None
    depth = _ZMAX if params.z_depth is None else params.z_depth
    return jnp.asarray(cache.z_top) == depth


def _pooled_curve(state: CompletionState, cache: PooledCountRatioCache, depth_ok=None):
    """The clipped pooled ratio on the redshift grid at this proposal's ``dN_exp``.

    NaN everywhere when ``depth_ok`` is ``False`` (:func:`_pooled_depth_ok`).
    """

    # z^2 / dN_exp is finite at z = 0, where the grid carries 0 / floor: the
    # first node takes the second's value (and no 0 / 0 enters reverse mode).
    u = zgrid**2 / jnp.where(zgrid > 0.0, state.dN_exp, 1.0)
    u = u.at[0].set(u[1])
    curve = jnp.clip(cache.operator @ u, 0.0, 1.0)
    if depth_ok is not None:
        curve = jnp.where(depth_ok, curve, jnp.nan)
    return curve


def _pooled_row_fraction(catalog: GalaxyCatalog, cache: PooledCountRatioCache, dtype):
    fraction = jnp.asarray(cache.row_fraction, dtype=dtype)
    n_rows = int(catalog.zgals.shape[0])
    if fraction.ndim != 1 or int(fraction.shape[0]) != n_rows:
        raise ValueError(
            "the pooled count-ratio cache must carry one row fraction per catalog "
            f"row, got {tuple(fraction.shape)} for {n_rows} rows"
        )
    return fraction


def _pooled_completion_curves(state, params, catalog, cache) -> CompletionCurves:
    """``C_p(z) = f_p clip[R(z)]`` on the grid, assembled as the row-fraction selection."""

    _require_cache_depth(cache, params)
    cbar = _pooled_curve(state, cache, _pooled_depth_ok(params, cache))
    fraction = _pooled_row_fraction(catalog, cache, cbar.dtype)
    C = fraction[:, None] * cbar[None, :]
    dN_exp = state.dN_exp[None, :]
    dN_miss = (1.0 - C) * dN_exp
    if params.z_depth is not None:
        dN_miss = jnp.where((zgrid <= params.z_depth)[None, :], dN_miss, dN_exp)
    N_miss = jnp.trapezoid(dN_miss, zgrid, axis=1)
    dN_exp_pos = jnp.where(state.dN_exp > 0.0, state.dN_exp, 1.0)[None, :]
    C_eff = jnp.clip(1.0 - dN_miss / dN_exp_pos, 0.0, 1.0)
    N_exp = jnp.trapezoid(state.dN_exp, zgrid)
    f = 1.0 - N_miss / jnp.where(N_exp > 0.0, N_exp, 1.0)
    return CompletionCurves(f=f, dN_miss=dN_miss, C_eff=C_eff, N_miss=N_miss, C=C)


def _build_completion_state_impl(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
) -> CompletionState:
    log_g = log_galaxy_measure_grid(cosmo, params)
    dN_exp = params.n0 * catalog.apix * jnp.exp(log_g)

    if params.z_depth is None:
        dN_exp_smooth = smoothing_operator() @ dN_exp
    else:
        depth_mask = zgrid <= params.z_depth
        dN_exp_smooth = smoothing_operator() @ jnp.where(
            depth_mask, dN_exp, 0.0
        )
        # Frozen gradient-safety spelling: the ratio is discarded beyond the
        # depth, so do not leave a vanishing denominator there for reverse mode.
        dN_exp_smooth = jnp.where(depth_mask, dN_exp_smooth, 1.0)

    return CompletionState(
        log_g_grid=log_g,
        dN_exp=dN_exp,
        dN_exp_smooth=dN_exp_smooth,
        z_depth=params.z_depth,
    )


@threads_distance_table()
def build_completion_state(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    distance_table=None,
) -> CompletionState:
    """Build the proposal-dependent ordinary completeness denominator grids."""

    return _build_completion_state_impl(cosmo, params, catalog)


def _row_completeness(
    row,
    state: CompletionState,
    catalog: GalaxyCatalog,
    observed_cache: ObservedDensityCache | None,
):
    dN_obs = (
        _observed_density_row(row, catalog)
        if observed_cache is None
        else observed_cache.dN_obs_kde[row]
    )
    safe = jnp.where(state.dN_exp_smooth > 0.0, state.dN_exp_smooth, 1.0)
    return jnp.clip(dN_obs / safe, 0.0, 1.0)


def _assemble_row(C, state: CompletionState):
    """Frozen ordinary non-LSS missing-count budget for one row."""

    dN_miss = (1.0 - C) * state.dN_exp
    if state.z_depth is not None:
        depth_mask = zgrid <= state.z_depth
        dN_miss = jnp.where(depth_mask, dN_miss, state.dN_exp)

    N_miss = jnp.trapezoid(dN_miss, zgrid)
    dN_exp_pos = jnp.where(state.dN_exp > 0.0, state.dN_exp, 1.0)
    C_eff = jnp.clip(1.0 - dN_miss / dN_exp_pos, 0.0, 1.0)
    N_exp = jnp.trapezoid(state.dN_exp, zgrid)
    f = 1.0 - N_miss / jnp.where(N_exp > 0.0, N_exp, 1.0)
    return f, dN_miss, C_eff, N_miss


def gathered_missing_density(curves: GatheredCompletionCurves, row, idx):
    """``dN_miss[row, idx]`` of the grid ``curves`` stands for (paired arrays)."""

    if curves.observed is not None:
        C = jnp.clip(curves.observed[row, idx] / curves.z_factor[idx], 0.0, 1.0)
    else:
        C = curves.row_fraction[row] * curves.z_factor[idx]
    dN_exp = curves.dN_exp[idx]
    dN_miss = (1.0 - C) * dN_exp
    if curves.depth_mask is not None:
        dN_miss = jnp.where(curves.depth_mask[idx], dN_miss, dN_exp)
    return dN_miss


#: Rows per block of the gathered ``N_miss`` reduction: its integrand is
#: (block x N_z), never (N_rows x N_z).  Each row's sum is the same in any block.
_MISSING_ROW_BLOCK: int = 256


def _missing_count_rows(rows, z_factor, dN_exp, depth_mask, observed_rows: bool):
    if observed_rows:
        C = jnp.clip(rows / z_factor[None, :], 0.0, 1.0)
    else:
        C = rows[:, None] * z_factor[None, :]
    dN_miss = (1.0 - C) * dN_exp[None, :]
    if depth_mask is not None:
        dN_miss = jnp.where(depth_mask[None, :], dN_miss, dN_exp[None, :])
    return jnp.trapezoid(dN_miss, zgrid, axis=-1)


def gathered_missing_count(
    *, observed=None, row_fraction=None, z_factor, dN_exp, depth_mask=None
):
    """Per-row ``N_miss``: the trapezoid of ``dN_miss`` over the redshift grid.

    The grid arithmetic row by row, in blocks of ``_MISSING_ROW_BLOCK`` rows,
    so the ``(N_rows, N_z)`` integrand is never formed at once.
    """

    observed_rows = observed is not None
    rows = observed if observed_rows else row_fraction
    n = int(rows.shape[0])
    block = _MISSING_ROW_BLOCK
    if n <= block:
        return _missing_count_rows(rows, z_factor, dN_exp, depth_mask, observed_rows)

    # Slice blocks in place (no padded copy of the rows); the last block is
    # shifted back to end at row n and rewrites rows it shares with the one
    # before with the same values.
    def body(i, out):
        start = jnp.minimum(i * block, n - block)
        part = jax.lax.dynamic_slice_in_dim(rows, start, block, axis=0)
        value = _missing_count_rows(part, z_factor, dN_exp, depth_mask, observed_rows)
        return jax.lax.dynamic_update_slice_in_dim(out, value, start, axis=0)

    dtype = jnp.result_type(rows.dtype, z_factor.dtype, dN_exp.dtype)
    return jax.lax.fori_loop(0, -(-n // block), body, jnp.zeros((n,), dtype))


def _gathered_completion_curves_impl(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    observed_cache: ObservedDensityCache,
) -> GatheredCompletionCurves:
    state = _build_completion_state_impl(cosmo, params, catalog)
    if isinstance(observed_cache, PooledCountRatioCache):
        # Pooled count ratio: the row-fraction factors.
        _require_cache_depth(observed_cache, params)
        cbar = _pooled_curve(state, observed_cache, _pooled_depth_ok(params, observed_cache))
        fraction = _pooled_row_fraction(catalog, observed_cache, cbar.dtype)
        depth_mask = None if params.z_depth is None else zgrid <= params.z_depth
        return GatheredCompletionCurves(
            observed=None,
            row_fraction=fraction,
            z_factor=cbar,
            dN_exp=state.dN_exp,
            depth_mask=depth_mask,
            N_miss=gathered_missing_count(
                row_fraction=fraction, z_factor=cbar, dN_exp=state.dN_exp,
                depth_mask=depth_mask,
            ),
        )
    safe = jnp.where(state.dN_exp_smooth > 0.0, state.dN_exp_smooth, 1.0)
    depth_mask = None if params.z_depth is None else zgrid <= params.z_depth
    observed = observed_cache.dN_obs_kde
    return GatheredCompletionCurves(
        observed=observed,
        row_fraction=None,
        z_factor=safe,
        dN_exp=state.dN_exp,
        depth_mask=depth_mask,
        N_miss=gathered_missing_count(
            observed=observed, z_factor=safe, dN_exp=state.dN_exp, depth_mask=depth_mask
        ),
    )


@threads_distance_table()
def gathered_completion_curves(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    observed_cache: ObservedDensityCache,
    distance_table=None,
) -> GatheredCompletionCurves:
    """:func:`completion_curves` kept as factors (``missing_density="gather"``).

    The count-ratio completeness of :func:`completion_curves`, with the
    ``(N_rows, N_z)`` ``C`` and ``dN_miss`` grids never formed: the prior
    reads ``dN_miss`` at each sample's two bracketing grid nodes from the
    observed-density cache (:func:`gathered_missing_density`), and ``N_miss``
    is reduced in row blocks.  Needs the cache (bind time).  The diagnostic
    ``f``, ``C`` and ``C_eff`` are not formed.
    """

    return _gathered_completion_curves_impl(cosmo, params, catalog, observed_cache)


def _completion_curves_impl(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    observed_cache: ObservedDensityCache | None,
) -> CompletionCurves:
    state = _build_completion_state_impl(cosmo, params, catalog)
    if isinstance(observed_cache, PooledCountRatioCache):
        return _pooled_completion_curves(state, params, catalog, observed_cache)
    rows = jnp.arange(catalog.zgals.shape[0], dtype=jnp.int32)
    C = vmap(
        lambda row: _row_completeness(row, state, catalog, observed_cache)
    )(rows)
    f, dN_miss, C_eff, N_miss = vmap(
        lambda curve: _assemble_row(curve, state),
        out_axes=(0, 0, 0, 0),
    )(C)
    return CompletionCurves(
        f=f,
        dN_miss=dN_miss,
        C_eff=C_eff,
        N_miss=N_miss,
        C=C,
    )


@threads_distance_table()
def completion_curves(
    cosmo: CosmologyParameters,
    params: CatalogParameters,
    catalog: GalaxyCatalog,
    observed_cache: ObservedDensityCache | None = None,
    distance_table=None,
) -> CompletionCurves:
    """Evaluate all ordinary non-LSS completeness curves for one proposal."""

    return _completion_curves_impl(cosmo, params, catalog, observed_cache)


__all__ = [
    "CompletionCurves",
    "CompletionState",
    "GatheredCompletionCurves",
    "ObservedDensityCache",
    "POOLED_BIN_WIDTH",
    "POOLED_WINDOW_DEFAULT",
    "PooledCacheDepth",
    "PooledCountRatioCache",
    "SIGMA_SMOOTH",
    "build_completion_state",
    "build_observed_density_cache",
    "build_pooled_count_ratio_cache",
    "bound_smoothing_operator",
    "completion_curves",
    "gathered_completion_curves",
    "gathered_missing_count",
    "galaxies_outside_row_fraction",
    "gathered_missing_density",
    "refuse_galaxies_outside_row_fraction",
    "smoothing_operator",
]
