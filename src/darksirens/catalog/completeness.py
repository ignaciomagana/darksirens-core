"""Ordinary non-LSS catalog completeness and missing-host count budget.

This module reconstructs only the frozen legacy per-row count-ratio estimator.
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

Opt-in (``ds.model(..., count_ratio="pooled")``): the pooled count ratio of
:func:`build_pooled_count_ratio_cache`, which smooths the ratio itself, pools
the catalog's rows before the clip and takes its kernel width as an argument.
The functions here dispatch on the cache they are given; the default cache
and its arithmetic are unchanged.
"""

from __future__ import annotations

import contextlib
import contextvars
import warnings
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


def build_observed_density_cache(
    catalog: GalaxyCatalog,
    *,
    batch_size: int = 512,
) -> ObservedDensityCache:
    """Precompute the theta-independent observed count KDE row-for-row.

    The even-split/padded batching is numerically inert and mirrors the mature
    legacy cache builder while using compact row ids directly.  No dense global
    pixel lookup is constructed.
    """

    n_rows = int(np.shape(catalog.zgals)[0])
    out = np.empty((n_rows, int(zgrid.size)), dtype=np.float64)
    if n_rows == 0:
        return ObservedDensityCache(jnp.asarray(out))

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


class PooledCountRatioCache(NamedTuple):
    """Theta-independent operands of the pooled count ratio (opt-in).

    ``operator`` is ``(N_z, N_z)``: applied to ``z^2 / dN_exp`` on the
    redshift grid it gives the pooled ratio ``R(z)`` of
    :func:`build_pooled_count_ratio_cache`.  ``row_fraction`` is the coverage
    fraction of each row of the catalog view the cache is used with.  It
    takes the place of :class:`ObservedDensityCache` wherever the count-ratio
    curves are evaluated.
    """

    operator: Any
    row_fraction: Any


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


def build_pooled_count_ratio_cache(
    catalog: GalaxyCatalog,
    *,
    window: float = POOLED_WINDOW_DEFAULT,
    z_depth: float | None = None,
    row_fraction=None,
) -> PooledCountRatioCache:
    """Precompute the pooled count ratio of a catalog (opt-in ``count_ratio="pooled"``).

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
    cell, the rows that ``row_fraction`` covers (every row without one), and
    a catalog whose depth varies over the sky has to be split into one
    catalog per depth (a field mixture) by the caller.

    ``dN_exp`` depends on the proposal, so what is stored is linear in it:
    ``1 / dN_exp(z_i)`` is ``z_i^-2`` times the linear interpolation of
    ``z^2 / dN_exp`` between the grid nodes, and the galaxies enter through
    their counts in redshift bins of ``POOLED_BIN_WIDTH``.  Real galaxies
    with ``z <= 0`` or ``z > z_top`` are not counted.  The catalog is read in
    row blocks; the cache is one ``(N_z, N_z)`` operator and one value per
    row, in place of the ``(N_rows, N_z)`` observed-density cache.

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
    if row_fraction is None:
        fraction = np.ones(n_rows, dtype=np.float64)
        n_empty = int(np.count_nonzero(ngals == 0))
        if n_empty > POOLED_EMPTY_ROW_WARN * n_rows:
            warnings.warn(
                f"count_ratio='pooled': {n_empty} of {n_rows} catalog rows hold no "
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
    total = float(fraction.sum())
    if not total > 0.0:
        raise ValueError("count_ratio='pooled' needs a positive summed row coverage")

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

    # Noise of the pooled ratio where the catalog is best sampled: the Kish
    # effective number of galaxies under one kernel at the median redshift.
    median = centres[np.searchsorted(np.cumsum(number), 0.5 * number.sum())]
    kernel = np.exp(-0.5 * ((median - centres) / window) ** 2) / centres**2
    effective = float(np.sum(number * kernel) ** 2 / np.sum(number * kernel**2))
    text = (
        f"count_ratio='pooled': a kernel of width {window:g} holds {effective:.0f} "
        f"effective galaxies at the catalog's median redshift {median:.3f} "
        f"({int(number.sum())} galaxies pooled), a relative noise of about "
        f"{1.0 / np.sqrt(effective):.2f} in the completeness"
    )
    if effective < POOLED_MIN_EFFECTIVE:
        raise ValueError(
            f"{text}; fewer than {POOLED_MIN_EFFECTIVE:g} is refused. Use a wider "
            "count_ratio_window, pool more rows, or completeness='selection'"
        )
    if effective < POOLED_WARN_EFFECTIVE:
        warnings.warn(
            f"{text}; below {POOLED_WARN_EFFECTIVE:g} the clip at 1 biases the "
            "completeness low. Consider a wider count_ratio_window",
            UserWarning,
            stacklevel=2,
        )

    # A[k, j] = sum_b G_w(z_k - c_b) (n_b / c_b^2) phi_j(c_b), phi_j the hat
    # function of node j: the galaxy sum with 1/dN_exp left as a vector.
    upper = np.clip(np.searchsorted(z, centres, side="right"), 1, z.size - 1)
    t = (centres - z[upper - 1]) / (z[upper] - z[upper - 1])
    operator = np.zeros((z.size, z.size), dtype=np.float64)
    for start in range(0, centres.size, 4096):
        part = slice(start, start + 4096)
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
        operator=jnp.asarray(operator), row_fraction=jnp.asarray(fraction)
    )


def _pooled_curve(state: CompletionState, cache: PooledCountRatioCache):
    """The clipped pooled ratio on the redshift grid at this proposal's ``dN_exp``."""

    u = zgrid**2 / jnp.where(state.dN_exp > 0.0, state.dN_exp, 1.0)
    # z^2 / dN_exp is finite at z = 0; the grid's first node carries 0 / floor.
    u = u.at[0].set(u[1])
    return jnp.clip(cache.operator @ u, 0.0, 1.0)


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

    cbar = _pooled_curve(state, cache)
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
        # Opt-in pooled count ratio: the row-fraction factors.
        cbar = _pooled_curve(state, observed_cache)
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
    "PooledCountRatioCache",
    "SIGMA_SMOOTH",
    "build_completion_state",
    "build_observed_density_cache",
    "build_pooled_count_ratio_cache",
    "bound_smoothing_operator",
    "completion_curves",
    "gathered_completion_curves",
    "gathered_missing_count",
    "gathered_missing_density",
    "smoothing_operator",
]
